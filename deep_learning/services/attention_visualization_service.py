import os
import pandas as pd
import numpy as np
import cv2
from image_visualization.image_fusion import ImageFusion
import shutil
from pathlib import Path


class ImageComposer():
    def __init__(self, tile_size, min_max_coords, data, intervals):
        self.height_units = min_max_coords['max_y']-min_max_coords['min_y']
        self.width_units = min_max_coords['max_x']-min_max_coords['min_x']
        self.assembly_width = (self.width_units+1)*tile_size
        self.assembly_height = (self.height_units+1)*tile_size
        self.tile_size = tile_size
        self.min_max_coords = min_max_coords
        length = len(data['coords'])
        self.data = [{key: data[key][i] for key in data}
                     for i in range(length)]
        self.intervals = intervals

    def create_composed_images(self, input_path, output_path, label):
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        for image_data in self.data:

            x = image_data['coords']['x']
            y = image_data['coords']['y']

            output_image = self.compose_single_image(
                input_path, (x, y), label,image_data[label])

            shifted_x = x-self.min_max_coords['min_x']
            shifted_y = y-self.min_max_coords['min_y']
            image_path = f"{shifted_x}_{shifted_y}.png"
            out_path = os.path.join(output_path, image_path)
            success = cv2.imwrite(out_path, output_image)

    def compose_single_image(self, input_path, coords, label, value):
        interpolated_value = interpolate_value(self.intervals[label], value)
        image_path = f"{coords[0]}_{coords[1]}.png"
        image = cv2.imread(f"{input_path}/{image_path}")
        height, width, _ = image.shape
        attention_image = create_attention_image(
            label, interpolated_value, height, width)
        return ImageFusion.add_images(image, attention_image, 0.5)


def create_attention_image(label, value, height, width):
    color = ATTENTION_COLORS[label]
    color = tuple(int(x * value) for x in color)
    return np.full((height, width, 3), color, dtype=np.uint8)


def interpolate_value(interval, value):
    min_val, max_val = interval
    normalized_value = (value - min_val) / (max_val - min_val)
    return max(0, min(1, normalized_value))


ATTENTION_COLORS = {
    0: (255, 165, 0),   # Blue
    1: (255, 255, 0),   # Cyan
    2: (0, 255, 0),     # Green
    3: (0, 255, 255),   # Yellow
    4: (0, 165, 255),   # Orange
    5: (0, 0, 255)      # Red
}
