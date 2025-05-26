import os
import numpy as np
import shutil
from tqdm import tqdm


"""
in order to create a dataset according to PyTorch ImageFolder format,
this script is used to separate images according to their corresponding
class label file

input folder structure:         output folder structure:
images/                         images/
|---image0.png                  |---angry/
|---image1.png                      |---image0.png
.                                   ...
.                               |---disgust/
.                                   |---image1.png
.                                   ...
.                               ...
.                               |---surprise/
|---image3455.pn                |---iamge3455.png
""" 

def divide_into_subfolders(labels_dir,images_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    class_names = {0: 'neutral',1: 'happy',2: 'sad',3: 'surprise',4: 'fear',\
                5: 'disgust', 6: 'anger',7: 'contempt'}

    for class_id in class_names:
        class_folder = os.path.join(output_dir,class_names[class_id])
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

    for file in tqdm(os.listdir(labels_dir),leave=True,unit='file'):
        if file.endswith('_exp.npy'):
            img_num = file[:-8]
            img_name = f"{img_num}.jpg"
            
            label_path = os.path.join(labels_dir,file)
            class_label = int(np.load(label_path).item())
            # print(f"Loaded label for file {file}: {class_label}")
            # print(f"Corresponding element: {class_names[class_label]}")
            if class_label in class_names.keys():
                
                dest_folder = os.path.join(output_dir,class_names[class_label])
            else:
                print(f"Unknown class {class_label} for file {file}")
                continue

            image_path = os.path.join(images_dir,img_name)
            if os.path.exists(image_path):
                shutil.move(image_path,dest_folder)
            else:
                print(f"No image found with the name {img_name}")
            
    print("Images have been organized into subfolders.")

    return 

def main():

    sets = ['train_set','val_set']
    
    for set in sets:

        images_dir = f'rest of path / {set}/images'
        labels_dir = f'rest of path /{set}/expression_labels'
        output_dir = f'rest of path / subf_{set}'

        divide_into_subfolders(labels_dir,images_dir,output_dir)

if __name__ == "__main__":
    main()