import os
import shutil
from tqdm import tqdm


"""
remove_contempt: function that moves folder of contempt images to a backup folder
can be parameterized for other emotions

add_gen_images: moves generated images to corresponding class subfolder
"""

def remove_contempt(source_dir,backup_dir):

    if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
    
    remove_emotions = ['contempt']
    
    for emotion in remove_emotions:
        print(emotion)
        backup_folder = os.path.join(backup_dir,emotion)
        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)

        rmv_folder = os.path.join(source_dir,emotion)
        for image in tqdm(os.listdir(rmv_folder)):
            image_path = os.path.join(rmv_folder,image)
            shutil.move(image_path,backup_folder)

    return

def add_gen_images(gen_dir,train_dir):

    for genfolder in os.listdir(gen_dir):
        train_folder = os.path.join(train_dir,genfolder)
        gen_folder_path = os.path.join(gen_dir,genfolder)
        for gen_image in tqdm(os.listdir(gen_folder_path)):
            gen_image_path = os.path.join(gen_folder_path,gen_image)
            shutil.copy(gen_image_path,train_folder)

    return 


def main():
     
    source_dir = "TRAIN_SET_DIR"
    backup_dir = "BACKUP_DIR"
    gen_dir = "GENERATED IMAGES PATH"

    # remove_neutral_contempt(source_dir,backup_dir)
    add_gen_images(gen_dir,source_dir)

if __name__ == "__main__":
    main()