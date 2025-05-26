import os
import sys
import shutil
from tqdm import tqdm
from instance_remove import remove_instances

"""
equal_instances: function that finds the class with min number of instances,
and returns a dict with how many instances need to be removed from other classes
in order to sub-sample and balance the dataset
"""
def equal_instances(src_dir:str) -> dict[str,int]:

    emotions = [f.path for f in os.scandir(src_dir) if f.is_dir]
    num_instances = {}
    for emotion in emotions:
        emotion_instances = len(os.listdir(emotion))
        num_instances[os.path.basename(emotion)] = emotion_instances

    print(num_instances)
    least_repr_emotion = min(num_instances.values())
    num_remove = {key:(value-least_repr_emotion) for key,value in num_instances.items()}   
    print(num_remove)

    return num_remove


def main():

    src_dir = 'SOURCE_DIR'
    backup_dir = 'BACKUP_DIR'

    reduc_emotions = equal_instances(src_dir)
    
    remove_instances(reduc_emotions,src_dir,backup_dir)

if __name__ == "__main__":
    main()
