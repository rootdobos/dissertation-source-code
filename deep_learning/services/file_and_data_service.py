import os
import torch


class FileDataService():
    def __init__(self, datadir):
        self.datadir = datadir

    def load_image_features(self, image_id):

        feature_path = os.path.join(self.datadir, "{}.pt".format(image_id))
        FileDataService.load_slide_features(feature_path)

    @staticmethod
    def load_slide_features(features_path):
        feature_vector = torch.load(features_path)

        return feature_vector.squeeze()
