import cv2
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import os
import numpy as np
import glob
class ImageFusion():
    def __init__(self,
                 inverse_zoom_level,
                 original_input_dir,
                 tile_size):
        self.inverse_zoom_level=inverse_zoom_level
        self.original_input_dir=original_input_dir
        self.tile_size=tile_size
        
    
    def fuse_images_from_directory(self,title,dir):
        img_slide=open_slide(os.path.join(self.original_input_dir ,f"{title}.tiff"))
        tiles= DeepZoomGenerator(img_slide,tile_size=self.tile_size,overlap=0,limit_bounds=True)

        cols,rows= tiles.level_tiles[len(tiles.level_tiles)-self.inverse_zoom_level]
        assembly_width=(cols+1)*self.tile_size
        assembly_height=(rows+1)*self.tile_size
        canvas=np.zeros((assembly_height,assembly_width,3),dtype=np.uint8)

        #files=glob.glob(f'{dir}/*')
        files= next(os.walk(dir))
        images=files[2]

        for image_path in images:
            coord=image_path.split('.')[0].split('_')
            x=int(coord[0])
            y=int(coord[1])

            image=cv2.imread(f"{dir}/{image_path}")

            # for c in range(0,3):
            #     canvas[y*self.tile_size:(y+1)*self.tile_size,x*self.tile_size:(x+1)*self.tile_size,c]=image[:,:,c]
            
            canvas[y*self.tile_size:(y+1)*self.tile_size,x*self.tile_size:(x+1)*self.tile_size]=image
        cv2.imwrite(f"{dir}/output.png",canvas)
    @staticmethod
    def add_images_from_path(img_path1,img_path2,output,alpha):
        img1=cv2.imread(img_path1)
        img2=cv2.imread(img_path2)
        output_data=ImageFusion.add_images(img1,img2,alpha)
        out_path=os.path.join(output,"output.png")
        success=cv2.imwrite(out_path,output_data)


    @staticmethod
    def add_images(img1,img2,alpha):
        output=img1*alpha +img2*(1-alpha)

        return np.clip(output,0,255).astype(np.uint8)




