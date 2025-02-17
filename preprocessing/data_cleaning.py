import os
import pandas as pd

def list_all_folder_in_directory(dir):
    folders= [entry.name for entry in os.scandir(dir) if entry.is_dir()]
    return folders

def clean_dataset_csv(datadir,input_csv,output_csv):
    existing_data=list_all_folder_in_directory(datadir)
    input_df=pd.read_csv(input_csv)

    filtered_df=input_df[input_df["image_id"].isin(existing_data)]
    filtered_df.set_index("image_id",inplace=True)
    filtered_df.to_csv(output_csv)
    return filtered_df


