import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import pickle
import pandas as pd

import kornia.augmentation as K

from scipy.ndimage import median_filter

from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.datasets import SSL4EOLBenchmark
from torchgeo.transforms import AugmentationSequential


def get_3d_knucklebone_kernel():  
  f = np.zeros(shape=(5,5,5), dtype=int)
  for i in range(5):
    for j in range(5):
      for k in range(5):
        sum = np.abs(i-2) + np.abs(j-2) + np.abs(k-2)
        if sum <= 2:
          f[i,j,k] = 1

  return f

def get_values_at(np_array, ids):
  values = []
  for i in ids:
    values.append(np_array[i[0], i[1], i[2]])
  return values


class RGB_density():

  def __init__(self,class_idx=7):

    assert class_idx > 0
    assert class_idx < 18
    all_classes=[ 0, 1, 5, 24, 37, 61, 111, 121, 122, 131, 141, 142, 143, 152, 176, 190, 195]
    self.class_idx = class_idx
    self.classes = [0,all_classes[self.class_idx]] #just 0 and current class
    self.counter = np.zeros(shape=(256,256,256), dtype=np.uint32)
    task = SemanticSegmentationTask.load_from_checkpoint('/content/drive/MyDrive/landsat/epoch=12-step=3585.ckpt')
    self.model = task.model
    self.has_counter_been_smoothed = False
    self.hsi_data = SSL4EOLBenchmark(root="./data",
                         split='test',
                         download=True,
                         sensor= "oli_sr",
                         product="cdl",
                         classes=self.classes,
                         transforms=AugmentationSequential(K.CenterCrop((224,224)),
                                                          data_keys=["image", "mask"]))
  
  def fit_counter(self):
    self.model.eval()
    
    with torch.no_grad():
      for sample in tqdm(self.hsi_data):
        hsi_img = sample['image']
        pred_probs= self.model(torch.div(hsi_img,255)).detach().cpu()
        pred_all_classes = torch.argmax(pred_probs,axis=1)
        pred_current_class = pred_all_classes == self.class_idx
        correct = torch.logical_and(pred_current_class,sample['mask'])
        correct_idxs = torch.argwhere(correct)
        for idx in correct_idxs:
          r = hsi_img[0,3,idx[0],idx[1]]
          g = hsi_img[0,2,idx[0],idx[1]]
          b = hsi_img[0,1,idx[0],idx[1]]
          self.counter[int(r.item()),int(g.item()),int(b.item())] += 1

  def save_counter(self, base_path='/content/drive/MyDrive/landsat/'):
    assert not self.has_counter_been_smoothed
    base_path=Path(base_path)
    full_path = base_path.joinpath(f'{self.class_idx}.pkl')
    with open(full_path,'wb') as f:
      pickle.dump(self.counter,f)

  def load_counter(self, base_path='/content/drive/MyDrive/landsat/'):
    base_path=Path(base_path)
    full_path = base_path.joinpath(f'{self.class_idx}.pkl')
    with open(full_path,'rb') as f:
      self.counter = pickle.load(f)
    self.has_counter_been_smoothed = False

  def smooth_counter(self):
    assert not self.has_counter_been_smoothed
    f = get_3d_knucklebone_kernel()
    smoothed_counter = median_filter(self.counter, footprint=f, mode='constant', cval=0)
    self.counter = smoothed_counter
    self.has_counter_been_smoothed = True

  def harmonic_mean_with(self, another_density):
    assert type(another_density) == RGB_density

    if not self.has_counter_been_smoothed:
      self.smooth_counter()

    if not another_density.has_counter_been_smoothed:
      another_density.smooth_counter()

    numerator = np.multiply(self.counter, another_density.counter)*2
    denominator = np.add(self.counter, another_density.counter)
    out = np.zeros(shape=numerator.shape, dtype=np.float32)
    np.divide(numerator, denominator, where=denominator!=0, out=out)
    return out

  def ratio_with(self, another_density):
    assert type(another_density) == RGB_density

    if not self.has_counter_been_smoothed:
      self.smooth_counter()

    if not another_density.has_counter_been_smoothed:
      another_density.smooth_counter()

    numerator = self.counter
    denominator = np.add(self.counter, another_density.counter)
    out = np.zeros(shape=numerator.shape, dtype=np.float32)
    np.divide(numerator, denominator, where=denominator!=0, out=out)
    return out

  def candidates_with(self, another_density):
    assert type(another_density) == RGB_density

    H = self.harmonic_mean_with(another_density)
    R = self.ratio_with(another_density)
    ids = np.argwhere(H) # only where both have values
    h = get_values_at(H, ids)
    r = get_values_at(R, ids)
    return pd.DataFrame({'ids':list(ids), 'h':h, 'r':r})