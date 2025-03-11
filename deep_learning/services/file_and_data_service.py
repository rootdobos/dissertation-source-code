import os
import torch

class FileDataService():
    def __init__(self,datadir):
        self.datadir=datadir

    def load_image_features(self,image_id):
        
        feature_path= os.path.join(self.datadir,"{}.pt".format(image_id))
        feature_vector=torch.load(feature_path)

        return feature_vector.squeeze()