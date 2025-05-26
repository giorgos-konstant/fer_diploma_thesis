import random
import sys
import numpy as np
import torch
from collections import Counter
sys.path.append('FACIAL_EXPRESSION_CLASSIFIER_MODEL_PATH')
from utils.fer_models import FEClassifier

"""
Complimentary algorithm that calculates emotion scores after model makes prediction on number of frames
"""

class InferenceAlgorithm:

    def __init__(self):

        self.emotions = ['anger','disgust','fear','happy','neutral','sad','surprise']
        self.decay = -0.1 #non detected emotions with previously non-zero scores "fade" over time 
        self.threshold = 0.5 #threshold for model confidence in emotion prediction
        self.current_vals = np.zeros(7,dtype=np.float32)
        self.model = FEClassifier(base='efficientnet')
        self.model.load_state_dict(torch.load("CLASSIFIER_MODEL_PATH"))
        self.model.to('cuda')
        self.model.eval()

    def new_scores(self,frame_buffer,batch_size):

        if frame_buffer.nelement() == 0:
            return
        
        change = [self.decay for _ in range(7)] #how much is added/subtracted from each emotion's score
        idx , avg_sm = self.batched_inference(frame_buffer,batch_size) #return dominant emotion class number & its softmax score 
        if avg_sm > self.threshold: 
            change[idx] = avg_sm*0.2 #predicted emotion gets increase instead of decay, as long as it surpasses the confidence threshold 
        else:
            change[4] = (1-avg_sm)*0.2 #if emotion avg_softmax is not high, emotion is predicted as neutral
        
        temp = self.current_vals 
        self.current_vals = [round(min(val + ch,1.0),2) if val+ch>=0 else 0.0 for val,ch in zip(temp,change)] #calculate new values, restricting between 0 and 1
        avg_preds_dict = {key:val for key,val in zip(self.emotions,self.current_vals)} # place values with emotion labels in dict
        sorted_dict = dict(sorted(avg_preds_dict.items(),key= lambda item: item[1],reverse=False)) #sort based on scores
        return sorted_dict 
    
    def batched_inference(self,frames,batch_size):
        
        with torch.no_grad():
            _,probs = self.model(frames) #model makes inference on 30 consecutive frames where face was detected
        probs = probs.detach().cpu().numpy()
        avg_softmax = np.sum(probs,axis=0) #average softmax scores for each
        avg_softmax /= batch_size
        label_preds = np.argmax(np.array(probs),axis=1) #most probable class from each frame
        
        counter = Counter(label_preds) #counts the instances of all predicted emotions
        dominate_emotion_elem = counter.most_common(1) #most predicted emotions and times it got predicted
        dominate_emotion = dominate_emotion_elem[0][0] #most predicted emotion
        print(self.emotions[dominate_emotion],avg_softmax[dominate_emotion])
        return dominate_emotion , avg_softmax[dominate_emotion] #returns most predicted emotion and its averaged softmax across 30 frames