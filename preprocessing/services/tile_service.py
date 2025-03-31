import os
from ..tile_extractor import TileExtractor

class TileService():

    @classmethod
    def extract_tiles(image_id,data_dir,output_dir):
        
        tile_extractor=TileExtractor(
        input_dir=data_dir,
        output_dir=output_dir,
        tile_size=256,
        inverse_zoom_level=2
        )
        tile_extractor.process_image(image_id)