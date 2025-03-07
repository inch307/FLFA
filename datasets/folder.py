import numpy as np
from torchvision.datasets import ImageFolder, DatasetFolder

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, transform=self.transform, target_transform=self.target_transform)
        self.loader = imagefolder_obj.loader
        
        # ImageFolder의 samples는 (path, label) 튜플 리스트입니다.
        samples = imagefolder_obj.samples
        if self.dataidxs is not None:
            samples = [samples[i] for i in self.dataidxs]
        
        # samples를 분리하여 self.data (이미지 경로)와 self.target (레이블 배열)로 저장합니다.
        self.data = [s[0] for s in samples]
        self.target = np.array([s[1] for s in samples])

    def __getitem__(self, index):
        path = self.data[index]
        target = self.target[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return sample, target

    def __len__(self):
        return len(self.data)
    
    @property
    def num_classes(self):
        return len(np.unique(self.target))