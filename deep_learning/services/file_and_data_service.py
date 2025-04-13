import os
import torch
import pandas as pd


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

    @staticmethod
    def get_visualized_tiles_response(dir):
        image_paths = os.listdir(dir)
        coords = FileDataService.get_tiles_coords(dir)
        min_max_coords = FileDataService.get_min_max_coordinates(
            {"coords": coords})
        columnCount = min_max_coords['max_x'] - min_max_coords['min_x']
        rowCount = min_max_coords['max_y'] - min_max_coords['min_y']
        return {
            "columnCount": int(columnCount),
            "rowCount": int(rowCount),
            "minX": int(min_max_coords['min_x']),
            "minY": int(min_max_coords['min_y']),
            "imagePaths": image_paths
        }

    @staticmethod
    def get_min_max_coordinates(data):
        df = pd.DataFrame(data['coords'])
        return {
            'min_x': df['x'].min(),
            'max_x': df['x'].max(),
            'min_y': df['y'].min(),
            'max_y': df['y'].max()
        }

    @staticmethod
    def get_tiles_coords(path):
        files = next(os.walk(path))
        images = files[2]
        coords = []
        for image_path in images:
            coord = image_path.split('.')[0].split('_')
            coords.append({
                "x": int(coord[0]),
                "y": int(coord[1])

            })
        return coords
