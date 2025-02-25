import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
class FeatureDataset(Dataset):
    def __init__(self,dataframe,datadir):
        self.dataframe= dataframe.reset_index(drop=True)
        self.datadir=datadir

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, index):
       image_id=self.dataframe["image_id"][index]
       isup_grade=self.dataframe["isup_grade"][index]

       feature_path= os.path.join(self.datadir,"{}.pt".format(image_id))
       feature_vector=torch.load(feature_path)

       return feature_vector.squeeze(),torch.tensor(isup_grade, dtype=torch.int64)
