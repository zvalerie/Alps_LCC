import math
import numpy as np
import random
from PIL import Image
 
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

class Compose(object):
    '''
        Identically return the transforms of img, dem and mask
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, dem = None, mask=None):
        for t in self.transforms:
            x, dem, mask = t(x, dem, mask)
        return x, dem, mask
    
class MyRandomRotation90(object):
    '''
        Random rotation by 90 degree counter-clockwise
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, dem, mask):        
        if random.random() < self.p:
            img = F.rotate(img, 90) 
            dem = F.rotate(dem, 90)            
            mask = F.rotate(mask, 90)          
        return img, dem, mask
    

class MyRandomHorizontalFlip(object):
    def __init__(self, p = 0.5):
        self.p = p
 
    def __call__(self, image, dem, mask):
        if random.random() < self.p:
            image = F.hflip(image)
            dem = F.hflip(dem)
            mask = F.hflip(mask)
        return image, dem, mask
    
class MyRandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5."""
    def __init__(self, p = 0.5):
        self.p = p
 
    def __call__(self, image, dem, mask):
        if random.random() < self.p:
            image = F.vflip(image)
            dem = F.vflip(dem)
            mask = F.vflip(mask)
        return image, dem, mask
    
class MinMaxScaler(object):
    '''
        Transform the input data to be in the range of [0, 1]
    '''
    def __init__(self, max, min):
        self.max = max
        self.min = min

    def __call__(self, dem):        
        normalized_dem = (dem - self.min) / (self.max - self.min)  
        return normalized_dem

class AbsoluteScaler(object):
    '''
        Transform the altitue values to absolute elevation per tile
    '''
    def __init__(self,):
        pass

    def __call__(self, dem):        
        scaled_dem = dem - min(dem)  
        return scaled_dem