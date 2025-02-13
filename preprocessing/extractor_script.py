import sys
from tile_extractor import TileExtractor
import pandas as pd
import os
from multiprocessing import Pool
from tqdm import tqdm
if __name__ == '__main__':
    args=sys.argv

    data_dir=args[1]
    output_dir=args[2]
    tile_size=int(args[3])
    izl=int(args[4])
    pool_size=int(args[5])
    df = pd.read_csv(os.path.join(data_dir,'train.csv'))

    tile_extractor=TileExtractor(
        input_dir=data_dir,
        output_dir=output_dir,
        tile_size=tile_size,
        inverse_zoom_level=izl
    )

    with Pool(processes=pool_size) as pool:
        res = list(
            tqdm(pool.imap(tile_extractor.process_image, list(df['image_id'])), total = len(df['image_id']))
        )
