import os
import shutil
import random
from collections import Counter
from tqdm import tqdm


"""
create a 80/10/10 train/test/validation split from the dataset

Input folder structure:         Output folder structure:
class-folders/                  new_split/
|---anger/                      |---train/
    |---image1.png                  |---anger/
    |---image2.png                      |--- ...
    ...                             |---disgust/
|---disgust/                            |--- ...
    |---image1.png                  ...
    |---image2.png              |---test/
    ...                             ....
|---happy/                      |---val/
    |---image1.png                  ....
    |---image2.png
    ...
...
"""

def create_new_split(train_dir,new_split_dir,train_ratio=0.8,val_ratio=0.1,test_ratio=0.1):

    if not os.path.exists(new_split_dir):
        os.makedirs(new_split_dir)
    all_images = []
    for subset_dir in [train_dir]:
        for class_name in os.listdir(subset_dir):
            class_dir = os.path.join(subset_dir,class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir,img_name)
                    all_images.append((img_path,class_name))
    
    random.shuffle(all_images)

    total_images = len(all_images)
    num_train = int(total_images*train_ratio)
    num_val = int(total_images*val_ratio)
    num_test = int(total_images*test_ratio)

    new_train_images = all_images[:num_train]
    new_val_images = all_images[num_train:num_train+num_val]
    new_test_images = all_images[num_train+num_val:]
    
    for subset in ['train','val','test']:
        subset_dir = os.path.join(new_split_dir,subset+'_set')
        os.makedirs(subset_dir,exist_ok=True)
        for class_name in set(class_name for _, class_name in all_images):
            os.makedirs(os.path.join(subset_dir,class_name),exist_ok=True)

    
    for img_path,class_name in tqdm(new_train_images,desc='Copying train images...',unit='file'):
        shutil.copy(img_path,os.path.join(new_split_dir,'train_set',class_name,os.path.basename(img_path)))
    for img_path,class_name in tqdm(new_val_images,desc='Copying validation images...',unit='file'):
        shutil.copy(img_path,os.path.join(new_split_dir,'val_set',class_name,os.path.basename(img_path)))
    for img_path,class_name in tqdm(new_test_images,desc='Copying test images...',unit='file'):
        shutil.copy(img_path,os.path.join(new_split_dir,'test_set',class_name,os.path.basename(img_path)))

    print("80-10-10 Split Created Successfully")

def main():

    train_dir = "TRAIN_SET_SOURCE_DIR"
    new_split_dir = "DESTINATION_DIR_FOR_TRAIN_TEST_VAL_SPLIT"
    create_new_split(train_dir,new_split_dir)

if __name__ == "__main__":
    main()