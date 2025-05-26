import torch.nn as nn
import torchvision.models as models

"""
file for Early Stopping when training stagnates, and Classifier model architecture
"""
class EarlyStopping:
  def __init__(self,patience=5,delta=1e-2):
    self.patience = patience
    self.delta = delta
    self.best_score = None
    self.best_score_epoch = 0
    self.early_stop = False
    self.counter = 0
    self.best_model_state = None

  def __call__(self,loss,model,epoch):

    score = -loss

    if self.best_score is None:
      self.best_score  = score
      self.best_score_epoch = epoch
      self.best_model_state = model.state_dict()
    elif score < self.best_score + self.delta:
      self.counter += 1
      if self.counter >= self.patience:
        self.early_stop = True
    else:
      self.best_score = score
      self.best_model_state = model.state_dict()
      self.counter = 0
      self.best_score_epoch = epoch

  def load_best_model(self,model):
    model.load_state_dict(self.best_model_state)
    print(f"Early Stopping. Weights loaded from best epoch {self.best_score_epoch}")

class FEClassifier(nn.Module):
  def __init__(self,base):
    super(FEClassifier,self).__init__()
    
    self.base_model = None

    if base == 'mobilenet':
      self.base_model = models.mobilenet_v3_large(weights=None)
      last_layer = 1280
      self.base_model.classifier[3] = nn.Linear(last_layer,7,bias=True)

    if base == 'resnet' :
      self.base_model = models.resnet50(weights=None)
      last_layer = 2048
      self.base_model.fc = nn.Linear(last_layer,7,bias=True)

    if base == 'densenet':
      self.base_model = models.densenet169(weights=None)
      last_layer = 1664
      self.base_model.classifier = nn.Linear(last_layer,7,bias=True)

    if base == 'efficientnet':
      self.base_model = models.efficientnet_b2(weights=None)
      last_layer = 1408
      self.base_model.classifier[1] = nn.Linear(last_layer,7,bias=True)

    if base == 'convnext':
      self.base_model = models.convnext_tiny(weights=None)
      last_layer = 768
      self.base_model.classifier[2] = nn.Linear(last_layer,7,bias=True)

    self.base_model_name = self.base_model.__class__.__name__

    self.softmax = nn.Softmax(dim=1)

  def forward(self,x):

    logits = self.base_model(x)
    probs = self.softmax(logits)
    return logits,probs