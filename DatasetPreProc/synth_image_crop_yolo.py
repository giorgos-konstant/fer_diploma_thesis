import cv2
from tqdm import tqdm
import os
from ultralytics import YOLO

"""
Crops generated images and keeps face ROI
also moves cropped images to new directory
"""

def crop_faces(input_dir,output_dir,detector):

    # mean_face_pixels = {"224": 0 ,"448": 0,"672":0} #this was used exploratorily to see
    # num_images = {"224":0,"448":0,"672":0}          #mean are of bounding box detected for the faces

    # for filename in tqdm(os.listdir(input_dir),leave=False):

    #     if filename.lower().endswith(('.png','.jpg','.jpeg')):
    #         filepath = os.path.join(input_dir,filename)

    #         image = cv2.imread(filepath)
    #         if image is None:
    #             print(f"Could not read image {filename}")
    #             continue

    #         for sz in num_images.keys() :
    #             if image.shape[0] == int(sz): num_images[sz] += 1

    # print(num_images)

    for filename in tqdm(os.listdir(input_dir),leave=False):

        if filename.lower().endswith(('.png','.jpg','.jpeg')):
            filepath = os.path.join(input_dir,filename)

            image = cv2.imread(filepath)
            if image is None:
                print(f"Could not read image {filename}")
                continue
            
            min_face_size = 112
            if image.shape[0] == 448: min_face_size = 112
            if image.shape[0] == 672: min_face_size = 224

            # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            faces = detector(image)
            
            base_name = os.path.basename(filepath)
            new_dir = output_dir+"/"+base_name

            if len(faces[0]) == 1: 
                x1,y1,x2,y2 = faces[0].boxes.xyxy[0].tolist()
                roi_img = image[int(y1):int(y2), int(x1):int(x2)]
                roi_rsz = cv2.resize(roi_img, (224,224),interpolation=cv2.INTER_CUBIC)

                cv2.imwrite(new_dir,roi_rsz)
                # mean_face_pixels[str(image.shape[0])] += max(abs(x2-x1),abs(y2-y1))/num_images[str(image.shape[0])]

            else:
                image_rsz = cv2.resize(image,(224,224),interpolation = cv2.INTER_CUBIC)
                cv2.imwrite(new_dir,image_rsz)

    # print("Mean Face Size:", mean_face_pixels)
    
    return 

def main():

    emotions = ['fear','disgust']
    base_input_dir = "UNCROPPED_GENERATED_IMAGES_SOURCE_DIR"
    base_output_dir = "CROPPED_GENERATED_IMAGES_DEST_DIR"
    
    face_detector = YOLO("PATH TO PYTORCH YOLO MODEL")
    for emotion in emotions:
        input_dir = base_input_dir+emotion
        output_dir = base_output_dir+emotion
        print(f"Emotion: {emotion}")
        crop_faces(input_dir,output_dir,face_detector)

    return


if __name__ == "__main__":
    main()
