import os
import numpy as np
from PIL import Image
from openslide import OpenSlide
import cv2

from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator

class TileExtractor():
    def __init__(self,
                 input_dir,
                 output_dir,
                 tile_size):
        self.input_dir=input_dir
        self.output_dir=output_dir
        self.tile_size=tile_size
    
    def process_image(self,idx, provider=None):
        image_subdir_path=os.path.join(self.output_dir,idx)
        if os.path.exists(image_subdir_path):
            return
        try:
            img_slide=open_slide(os.path.join(self.input_dir,"train_images" ,f"{idx}.tiff"))
            mask_slide=open_slide(os.path.join(self.input_dir, "train_label_masks",f"{idx}_mask.tiff"))
            if not os.path.exists(image_subdir_path):
                os.mkdir(image_subdir_path)
            passed_coords=self.pyramid_tile_extracting(img_slide,8,220,210)
            self.save_tiles_from_coordinates(img_slide,
                                            mask_slide,
                                                passed_coords,
                                            image_subdir_path,
                                            provider)
        except:
            print(f"Error in extracting {idx}")
    def save_tiles_from_coordinates(self, slide,mask,coords,saving_dir,provider):
        tiles_zoom= DeepZoomGenerator(slide,tile_size=self.tile_size,overlap=0,limit_bounds=True)
        mask_zoom= DeepZoomGenerator(mask,tile_size=self.tile_size,overlap=0,limit_bounds=True)
        for c in coords:
            self.save_tile(tiles_zoom,c,os.path.join(saving_dir,"tiles"))
            self.save_tile(mask_zoom,c,os.path.join(saving_dir,"mask"),grayscale=provider)

    
    def save_tile(self, tiles,coord,saving_dir, grayscale=None):
        if not os.path.exists(saving_dir):
            os.mkdir(saving_dir)
        temp_tile=tiles.get_tile(len(tiles.level_tiles)-1,coord)
        temp_tile_np = np.array(temp_tile)

        if grayscale=="karolinska":
            temp_tile_np=temp_tile_np[:,:,0] *127
        elif grayscale=="radboud":
            temp_tile_np=temp_tile_np[:,:,0] *51
            
        im = Image.fromarray(temp_tile_np)
        im.save(os.path.join(saving_dir, f"{coord[0]}_{coord[1]}.png"), format='PNG', quality=100)


    def pyramid_tile_extracting(self, slide,down_sampling_factor, avg_threshold,median_threshold):
        """
        On every pyramid level the base image is smaller, but the sizes of the tiles are the same
        (for example 256), so this is why we need to create an image pyramid with smaller tile size,
        if we want to made statistical operation on smaller image and project it to the larger image.
        We need to find which level has the same rows and columns with the smaller and larger tile parameters,
        and we can succesfully project the calculations
        """
        tiles= DeepZoomGenerator(slide,tile_size=self.tile_size,overlap=0,limit_bounds=True)
        down_sampled_tiles= DeepZoomGenerator(slide,tile_size=self.tile_size/down_sampling_factor,overlap=0,limit_bounds=True)
        
        # EXPLANATION CODE
        # for subs in range(5):
        #     zoom_level=len(tiles.level_tiles)-1-subs
        #     tiles_shape= tiles.level_tiles[zoom_level]
        #     cols, rows= tiles_shape
        #     print(f" ORIGINAL Cols: {cols} Rows: {rows} Shape of tile: {tiles.get_tile(zoom_level,(0,0))}")

        #     zoom_level=len(down_sampled_tiles.level_tiles)-1-subs
        #     tiles_shape= down_sampled_tiles.level_tiles[zoom_level]
        #     cols, rows= tiles_shape
        #     print(f" DOWNSAMPLED Cols: {cols} Rows: {rows} Shape of tile: {down_sampled_tiles.get_tile(zoom_level,(0,0))}")
        # EXPLANATION CODE END

        tiles_shape= tiles.level_tiles[len(tiles.level_tiles)-1]
        cols, rows= tiles_shape
        down_tile_level_corresponding_level=down_sampled_tiles.level_tiles.index(tiles_shape)

        passed_coords=[]
        for col in range(1,cols-1):
            for row in range(1,rows-1):
                temp_tile=down_sampled_tiles.get_tile(down_tile_level_corresponding_level,(col,row))
                mean,median,std=self.tile_statistics(temp_tile)
                if mean<avg_threshold:
                    passed_coords.append((col,row))
        return passed_coords



    def tile_statistics(self,tile):
        tile_RGB=tile.convert('RGB')
        tile_np=np.array(tile_RGB)
        mean_colors=np.mean(tile,axis=(0,1))
        mean_img=np.mean(mean_colors)
        gray=cv2.cvtColor(tile_np,cv2.COLOR_RGB2GRAY)
        median_img=np.median(gray)
        std=np.std(gray)
        return mean_img,median_img,std