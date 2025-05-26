import os
import shutil
from tqdm import tqdm

"""
this is used after extraction of the original AffectNet
to move the valdation images to the train set 
to split them again later
"""

source_dir = 'SOURCE_DIR'
dest_dir = 'DEST_DIR'

for folder_name in os.listdir(source_dir):
    print(folder_name)
    src_folder_path = os.path.join(source_dir,folder_name)
    dest_folder_path = os.path.join(dest_dir,folder_name)

    for file_name in tqdm(os.listdir(src_folder_path)):
        src_file_path = os.path.join(src_folder_path,file_name)

        if os.path.isfile(src_file_path):
            new_name = f'val_{os.path.basename(src_file_path)}'

            dest_file_path = os.path.join(dest_folder_path,new_name)

            shutil.move(src_file_path,dest_file_path)
    
    print("Files transferred successfully from val to train set")