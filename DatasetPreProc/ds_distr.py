import os
import shutil
import matplotlib.pyplot as plt

"""
this file outputs a bar chart to visualize
the distribution of the dataset
"""

ds_dirs = ["TRAIN_SET_PATH",
          "TEST_SET_PATH",
          "VALIDATION_SET_PATH"]  #paths for the train, test, validation sets are located

plt.figure(figsize=(12,8))

prev = [0 for _ in range(7)] #this is useful for creating the stacked bar chart effect
y_triplets = [] 
for subset in ds_dirs:
    emotions = [f.path for f in os.scandir(subset) if f.is_dir()]
    y_axis = []
    x_axis = []
    for emotion in emotions:
        emotion_instances = len(os.listdir(emotion)) #count how many images are in each class in each set
        y_axis.append(emotion_instances)
        x_val = os.path.basename(emotion)
        x_axis.append(x_val)

    y_triplets.append(y_axis) 
    
    barplot = plt.bar(x_axis,y_axis,bottom=prev)

    prev = [sum(x) for x in zip(prev,y_axis)] 

y_labels = [f"{a}/\n{b}/\n{c}" for a,b,c in zip(*y_triplets)] #create the labels (number of instances) as triplets, less label occlusions that way

plt.bar_label(barplot,labels=y_labels,label_type='edge',fontsize=10,weight='bold',padding=20)

plt.ylim((0,175000))
plt.xlabel("Emotions")
plt.ylabel("Instances")
plt.title("Dataset Distribution")
plt.legend(["Train","Test","Validation"],bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.show()
