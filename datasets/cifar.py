from PIL import Image
import torchvision.datasets as datasets
from torch.utils.data import Dataset


class CIFAR10_(datasets.CIFAR10):
    """
    CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.transform is not None:
            img1 = self.transform(img)
            if self.train:
                img2 = self.transform(img)

        if self.train:
            return img1, img2, target, index
        else:
            return img1, target, index


class CIFAR20_(datasets.CIFAR100):
    """
    CIFAR20 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        target = cifar100_to_cifar20(target)
        img = Image.fromarray(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            img1 = self.transform(img)
            if self.train:
                img2 = self.transform(img)

        if self.train:
            return img1, img2, target, index
        else:
            return img1, target, index

class CIFAR100_(datasets.CIFAR100):
    """
    CIFAR100 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            img1 = self.transform(img)
            if self.train:
                img2 = self.transform(img)

        if self.train:
            return img1, img2, target, index
        else:
            return img1, target, index        
        
def cifar100_to_cifar20(target):
    """
    CIFAR100 to CIFAR 20 dictionary. 
    This function is from IIC github.
    """
  
    class_dict = {0: 4,
     1: 1,
     2: 14,
     3: 8,
     4: 0,
     5: 6,
     6: 7,
     7: 7,
     8: 18,
     9: 3,
     10: 3,
     11: 14,
     12: 9,
     13: 18,
     14: 7,
     15: 11,
     16: 3,
     17: 9,
     18: 7,
     19: 11,
     20: 6,
     21: 11,
     22: 5,
     23: 10,
     24: 7,
     25: 6,
     26: 13,
     27: 15,
     28: 3,
     29: 15,
     30: 0,
     31: 11,
     32: 1,
     33: 10,
     34: 12,
     35: 14,
     36: 16,
     37: 9,
     38: 11,
     39: 5,
     40: 5,
     41: 19,
     42: 8,
     43: 8,
     44: 15,
     45: 13,
     46: 14,
     47: 17,
     48: 18,
     49: 10,
     50: 16,
     51: 4,
     52: 17,
     53: 4,
     54: 2,
     55: 0,
     56: 17,
     57: 4,
     58: 18,
     59: 17,
     60: 10,
     61: 3,
     62: 2,
     63: 12,
     64: 12,
     65: 16,
     66: 12,
     67: 1,
     68: 9,
     69: 19,
     70: 2,
     71: 10,
     72: 0,
     73: 1,
     74: 16,
     75: 12,
     76: 9,
     77: 13,
     78: 15,
     79: 13,
     80: 16,
     81: 19,
     82: 2,
     83: 4,
     84: 6,
     85: 19,
     86: 5,
     87: 5,
     88: 8,
     89: 19,
     90: 18,
     91: 1,
     92: 2,
     93: 15,
     94: 6,
     95: 0,
     96: 17,
     97: 8,
     98: 14,
     99: 13}
    
    return class_dict[target]

