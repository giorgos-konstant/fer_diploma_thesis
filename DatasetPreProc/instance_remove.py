import os
import shutil
import random
from tqdm import tqdm

"""
Script to remove chosen number of instances of chosen emotions
initially it was used to reduce neutral and happy instances
in the end used in equal_instances.py file to balance the whole dataset
"""

def remove_instances(emotions : dict[str,int],
                     source_dir : str,
                     backup_dir : str):
    
    for emotion, num_rmv in emotions.items():
        class_dir = os.path.join(source_dir,emotion)
        files = os.listdir(class_dir)
        files_rmv = random.sample(files,num_rmv)

        for file in tqdm(files_rmv,desc=f'Removing {num_rmv} instances from {emotion}',unit='file'):
            
            file_path = os.path.join(class_dir,file)
            class_backup_dir = os.path.join(backup_dir,emotion)
            os.makedirs(class_backup_dir,exist_ok = True)
            shutil.move(file_path,class_backup_dir)

    return


def main():

    reduc_emotions = {'neutral' :  40000}
    source = 'SOURCE_DIR'
    backup = 'BACKUP_DIR'

    remove_instances(reduc_emotions,source,backup)

if __name__ == "__main__":
    main()