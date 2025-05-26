import os
import shutil
from tqdm import tqdm

"""
initially, files for valence, arousal and class label are all in the same folder
this moves the class label files to another folder

Input folder structure:         Output folder structure:
annotations/                    annotations/
|--- image0_exp.npy             |--- image0_arousal.npy
|--- image0_arousal.npy         |--- image0_valence.npy
|--- image0_valence.npy         ...
...                             expression_labels/
                                |--- image0_exp.npy
                                ...
""" 

def isolate_expression_labels(source_dir,dest_dir):

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    files = os.listdir(source_dir)

    for file in tqdm(files,leave=False,unit='file'):
        if 'exp' in file:
            src_path = os.path.join(source_dir,file)
            dst_path = os.path.join(dest_dir,file)
            shutil.copy(src_path,dst_path)

    print('Files copied successfully')

    return

def main():

    init_train_dir = ['TRAIN_SET_ANNOTATIONS_DIR','TRAIN_SET_EXPRESSIONS_DIR']
    init_val_dir = ['VAL_SET_ANNOTATIONS_DIR','VAL_SET_ANNOTATIONS_DIR']

    for dir in [init_train_dir,init_val_dir]:
        isolate_expression_labels(source_dir=dir[0],dest_dir=dir[1])

if __name__ == "__main__":
    main()