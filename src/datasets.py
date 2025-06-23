import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.tv_tensors as TVT
from lightning import LightningDataModule


class Rare25Dataset(Dataset):
    def __init__(self, dataset, idx_list, preprocessing=None, transform=None):
        self.dataset = dataset
        self.idx_list = idx_list
        self.preprocessing = preprocessing
        self.transform = transform
    
    def __len__(self):
        return len(self.idx_list)
    
    def __getitem__(self, index):
        # Load image and convert to torch tensor
        image = TVT.Image(self.dataset[self.idx_list[index]]['image'])
        label = self.dataset[self.idx_list[index]]['label']
        if self.preprocessing:
            image = self.preprocessing(image)
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def visualize(self, index):
        image, label = self[index]
        plt.figure(figsize=(3, 3))
        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()


class Rare25DataModule(LightningDataModule):
    def __init__(self,
                 split_ratio=(0.7, 0.1, 0.2),
                 preprocessing=None,
                 train_transform=None,
                 test_transform=None,
                 batch_size=16,
                 num_workers=4,
                 random_seed=666):
        super().__init__()
        self.dataset = load_dataset("TimJaspersTue/RARE25-train", split="train")
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing = preprocessing
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.train_sampler = None
        self.random_seed = random_seed
    
    def setup(self, stage=None):
        train_index, val_index, test_index = self._split()
        
        train_labels = np.array(self.dataset['label'])[train_index]
        train_labels_weight = train_labels * 9 + 1
        self.train_sampler = WeightedRandomSampler(train_labels_weight, 
                                                   num_samples=len(train_index), 
                                                   replacement=True)
        self.train_ds = Rare25Dataset(self.dataset, train_index,
                                      preprocessing=self.preprocessing,
                                      transform=self.train_transform,
                                      )
        self.val_ds = Rare25Dataset(self.dataset, val_index,
                                    preprocessing=self.preprocessing,
                                    transform=self.test_transform,
                                    )
        self.test_ds = Rare25Dataset(self.dataset, test_index,
                                     preprocessing=self.preprocessing,
                                     transform=self.test_transform,
                                     )
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, sampler=self.train_sampler, 
                          shuffle=True if self.train_sampler is None else False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
    def _split(self):
        train_ratio, val_ratio, test_ratio = self.split_ratio
        all_index = np.arange(len(self.dataset))
        labels = self.dataset['label']

        train_val_index, test_index = train_test_split(
            all_index, test_size=test_ratio, random_state=self.random_seed, stratify=labels
        )
        train_val_labels = [labels[i] for i in train_val_index]
        train_index, val_index = train_test_split(
            train_val_index,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=self.random_seed,
            stratify=train_val_labels,
        )
        assert len(train_index) + len(val_index) + len(test_index) == len(all_index)
        assert len(set(train_index).intersection(set(val_index))) == 0
        assert len(set(train_index).intersection(set(test_index))) == 0
        assert len(set(val_index).intersection(set(test_index))) == 0
        return train_index.tolist(), val_index.tolist(), test_index.tolist()

        