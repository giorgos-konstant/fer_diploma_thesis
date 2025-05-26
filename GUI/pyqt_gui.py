from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QImage,QPixmap, QFont
from ultralytics import YOLO
import pyqtgraph as pg
import torch
import torchvision.transforms.v2 as transforms
import sys
import cv2
import numpy as np
from datetime import datetime
import os
from time import time
sys.path.append('PATH_TO_INFERENCE_ALGORITHM_FILE')
from inference_algorithm.inference_algorithm import InferenceAlgorithm

"""
FILE FOR GUI IMPLEMENTATION
"""

class VideoWidget(QLabel):

    frame_cap = pyqtSignal(torch.Tensor) #makes frames available for bar plot widget

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(0) #camera feed
        self.detector = YOLO("FACE_DETECTOR_MODEL_PATH") #YOLO face detector
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.frame_buffer = []
        self.transform = transforms.Compose([transforms.ToImage(),
                                             transforms.ToDtype(torch.float32,scale=True),
                                             transforms.Normalize(mean = [0.5474, 0.4259, 0.3695], std = [0.2782, 0.2465, 0.2398])])
        
    def update_frame(self):
        ret, frame = self.cap.read() #capture frames
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width

            faces = self.detector(frame,verbose=False) #detect face from camera feed

            if len(faces[0]) == 1: # if 1 face is found
                x1,y1,x2,y2 = faces[0].boxes.xyxy[0].tolist() # get coordinates of top left corner and bottom right corner
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),1) #place rectangle on face ROI
            qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888) #sets data for the output seen on GUI (camera feed+face bounding box)
            self.setPixmap(QPixmap.fromImage(qimg))
            frame_roi = frame[int(y1)+2:int(y2), int(x1)+2:int(x2)] #keep face ROI from frame
            frame_rgb = cv2.cvtColor(frame_roi,cv2.COLOR_BGR2RGB)
            frame_rsz = cv2.resize(frame_rgb, (224,224),interpolation=cv2.INTER_CUBIC) #resize so it has appropriate input size for FER model (3,224,224)
            self.frame_buffer.append(self.transform(frame_rsz).to('cuda'))
            self.check_emit()

        return 
    
    def check_emit(self):
        if len(self.frame_buffer) == 30:
                self.tensors_buffer = torch.stack(self.frame_buffer,dim=0)
                self.frame_cap.emit(self.tensors_buffer) #emits the frame to listening events
                self.frame_buffer = []
            
        return 
            
    def closeEvent(self, event):
        self.cap.release()

class BarPlotWidget(pg.PlotWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackground('w')
        self.font = QFont()
        self.font.setPointSize(15)
        self.plotItem = self.getPlotItem() #plot for emotion prediction score table
        self.bar_graph_item = pg.BarGraphItem(x0=0,height=0.3,width=[],brush='skyblue')
        self.plotItem.addItem(self.bar_graph_item)
        self.setPlotBasics()
        self.buffer_size = 30
        self.helper = InferenceAlgorithm()
        self.bar_values = [pg.TextItem(text = ' ', anchor = (0,0.5),color = 'black') for _ in range(7)] 
        for value in self.bar_values:
            value.setFont(self.font)
            self.plotItem.addItem(value) #places values at edges of bars

    def update_plot(self,tensors_buffer=None):
        categories = self.helper.emotions 
        if tensors_buffer is not None:
            
            new_dict = self.helper.new_scores(tensors_buffer,self.buffer_size) #returns dict with new scores

            y = np.arange(len(categories))
            self.bar_graph_item.setOpts(y=y,width=list(new_dict.values()))
            for i,value in enumerate(new_dict.values()):
                self.bar_values[i].setPos(value,i)
                self.bar_values[i].setText(str(value))
            x_ticks = list(range(len(categories)))
            self.plotItem.getAxis('left').setTicks([list(zip(x_ticks, list(new_dict.keys())))])
        
        return
    
    def setPlotBasics(self):
        self.plotItem.setLabel(axis='bottom',text='Score',**{'font-size':'15pt'})
        self.plotItem.setLabel(axis='left',text='Emotions',**{'font-size':'15pt'})
        self.plotItem.setTitle('Emotion Prediction Score',size='15pt',color='black')
        self.plotItem.getAxis('bottom').setTickFont(self.font)
        self.plotItem.getAxis('left').setTickFont(self.font)
        self.plotItem.getAxis('bottom').setPen(**{"color":"black"})
        self.plotItem.getAxis('left').setPen(**{"color":"black"})
        self.plotItem.getAxis('bottom').setTextPen(**{"color":"black"})
        self.plotItem.getAxis('left').setTextPen(**{"color":"black"})
        self.plotItem.setXRange(0,1) #restrict scores between 0 and 1
        return

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setGeometry(100,100,1200,800)
        self.setStyleSheet('background : lightgrey;')

        layout = QHBoxLayout()

        self.video_widget = VideoWidget(self)
        layout.addWidget(self.video_widget)

        self.bar_plot_widget = BarPlotWidget(self)
        layout.addWidget(self.bar_plot_widget)

        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.video_widget.frame_cap.connect(self.bar_plot_widget.update_plot) #this allows "transmission" of frame data across widgets
        
        self.setWindowTitle("Diploma Thesis: Facial Expression Recognition GUI")
        self.show()

if __name__ == "__main__" :
    App = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(App.exec())
