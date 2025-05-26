import time
import sklearn
import sklearn.manifold
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import random
import torch.nn as nn
import torchvision.models as models
from torcheval.metrics.functional import multiclass_accuracy, multiclass_auroc, multiclass_f1_score, multiclass_precision, multiclass_recall
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from datetime import datetime, timedelta
import json
from sklearn.model_selection import KFold

"""
this file contains all functions necessary for data preparation, training, testing, plotting and saving results
"""

#create Dataset instances from pertaining subset folders
def create_dataset(train_dir : str,val_dir : str,test_dir : str):

    train_transforms_list = [transforms.RandomHorizontalFlip(0.3),
                                    transforms.RandomRotation(degrees=(0,20)),
                                    transforms.ColorJitter(brightness=0.3,hue=0.15),
                                    transforms.RandomAdjustSharpness(sharpness_factor=2),
                                    transforms.RandomAutocontrast()]

    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                            transforms.RandomApply(train_transforms_list,p=0.5),
                                            transforms.ToImage(),
                                            transforms.ToDtype(torch.float32,scale=True),
                                            transforms.Normalize(mean = [0.5474, 0.4259, 0.3695], std = [0.2782, 0.2465, 0.2398])])

    valid_test_transform = transforms.Compose([transforms.Resize((224,224)),
                                              transforms.ToImage(),
                                              transforms.ToDtype(torch.float32,scale=True),
                                              transforms.Normalize(mean = [0.5474, 0.4259, 0.3695], std = [0.2782, 0.2465, 0.2398])])
        
    train_dataset = datasets.ImageFolder(root=train_dir,transform=train_transforms)
    val_dataset = datasets.ImageFolder(root=val_dir,transform=valid_test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir,transform=valid_test_transform)
       
    total_images = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print(f"Train-Validation-Test Split {len(train_dataset):,}-{len(val_dataset):,}-{len(test_dataset):,}".replace(",","."))
    print(f"Train-Validation-Test % Split {int(len(train_dataset)*100/(total_images))}-{int(len(val_dataset)*100/total_images)}-{int(len(test_dataset)*100/total_images)}")

    return train_dataset,val_dataset,test_dataset

#create DataLoader instances for model input
def create_data_loaders(train_dataset, val_dataset, test_dataset,batch_size : int):

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=8,pin_memory=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True)
    print(f"Train-Validation-Test Split Batches {len(train_loader)}-{len(val_loader)}-{len(test_loader)}")
    
    return train_loader,val_loader,test_loader

#create concatenated dataset (train+validation) and test loader
def create_kfold_structs(train_dataset,val_dataset,test_dataset,batch_size : int):
   
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=8,pin_memory=True)
    train_val_dataset = torch.utils.data.ConcatDataset([train_dataset,val_dataset])
    return train_val_dataset,test_loader

#print each emotion instances in each subset
def dataset_distribution(train_dataset, val_dataset, test_dataset) :

    class_names = {v:k for k,v in train_dataset.class_to_idx.items()} #reverse the notation of num: "emotion" to "emotion" : num
    subsets = [train_dataset,val_dataset,test_dataset] 
    names = ['Train Set','Valid Set','Test Set']
    for i,subset in enumerate(subsets):
        instances = np.unique(subset.targets, return_counts=True) #count instances of each class in each subset
        print(names[i]) 
        class_distr = {}
        for k,v in zip(class_names.values(),instances[1]):
            class_distr[k] = v #save counts to a emotion-specific key
        print(class_distr)

    return

#save training metrics,hyperparameters in xlsx file
def save_to_excel(df : pd.DataFrame , excel_file: str):

    with pd.ExcelWriter(excel_file,engine='xlsxwriter') as writer:
        df.to_excel(writer,index=False,sheet_name='Training Data')

        workbook = writer.book
        worksheet = writer.sheets['Training Data']

        number_format = workbook.add_format({'num_format': '0.0000000'}) #fixes number format so its 3 decimal digits

        for col in df.columns:
            col_idx = df.columns.get_loc(col) + 1
            worksheet.set_column(col_idx,col_idx,None,number_format) 

        worksheet.autofit() # autofit columns
        print("Saved training data to excel successfully.")

    return
  
#calculate classification metrics [ACC,F1,PREC,REC,AUROC]
def calculate_metrics(pred,true,probs):

  acc = multiclass_accuracy(pred,true,num_classes=7) 
  f1_score = multiclass_f1_score(pred,true,num_classes=7,average='macro') #weighted -> calculate for each separately, weighted sum depeding on number of instances 
  prec = multiclass_precision(pred,true,num_classes=7,average='macro')    #if instances are equal weighted == macro average
  rec = multiclass_recall(pred,true,num_classes=7,average='macro')
  auroc = multiclass_auroc(probs,true,num_classes=7,average='macro')
  return acc,f1_score,prec,rec,auroc

# metric values by epoch are stored in a dict where KEYS are the name of the metric and VALUES are lists,
# so at the end of each epoch, the new value gets appended,
# values are needed for later plotting for model training performance
def append_metrics(case:str,metric_dict: dict,loss: float,acc: float, f1_score: float, prec: float, rec: float, auroc: float):

    if case == 'validation':
        metric_dict['val_loss'].append(loss)
        metric_dict['val_acc'].append(acc)
        metric_dict['val_f1_score'].append(f1_score)
        metric_dict['val_precision'].append(prec)
        metric_dict['val_recall'].append(rec)
        metric_dict['val_auroc'].append(auroc)
        return

    metric_dict['train_loss'].append(loss)
    metric_dict['acc'].append(acc)
    metric_dict['f1_score'].append(f1_score)
    metric_dict['precision'].append(prec)
    metric_dict['recall'].append(rec)
    metric_dict['auroc'].append(auroc)

    return

#basic logging and printing of losses,metrics and hyperparameters after each training
#you can never be too safe with storing data you might need for later analysis
def log_print_info(epoch,train_loss,val_loss,lr,epoch_dur,metric_dict):

    logging.info(f"Epoch {epoch}: Training Loss = {train_loss:.3f} Validation Loss = {val_loss:.3f} LR = {lr:.1e} Duration = {str(epoch_dur)[2:7]} mm:ss")
    logging.info(f"f1_score = {metric_dict['f1_score'][-1]:.3f} val_f1_score = {metric_dict['val_f1_score'][-1]:.3f} acc = {metric_dict['acc'][-1]:.3f} val_acc = {metric_dict['val_acc'][-1]:.3f} prec = {metric_dict['precision'][-1]:.3f} val_prec = {metric_dict['val_precision'][-1]:.3f} rec = {metric_dict['recall'][-1]:.3f} val_rec = {metric_dict['val_recall'][-1]:.3f} auroc = {metric_dict['auroc'][-1]:.3f} val_auroc = {metric_dict['val_auroc'][-1]:.3f}")
    print(f"Epoch {epoch}: Training Loss = {train_loss:.3f} Validation Loss = {val_loss:.3f} LR = {lr:.1e} Duration = {str(epoch_dur)[2:7]} mm:ss")
    print(f"f1_score = {metric_dict['f1_score'][-1]:.3f} val_f1_score = {metric_dict['val_f1_score'][-1]:.3f} acc = {metric_dict['acc'][-1]:.3f} val_acc = {metric_dict['val_acc'][-1]:.3f} prec = {metric_dict['precision'][-1]:.3f} val_prec = {metric_dict['val_precision'][-1]:.3f} rec = {metric_dict['recall'][-1]:.3f} val_rec = {metric_dict['val_recall'][-1]:.3f} auroc = {metric_dict['auroc'][-1]:.3f} val_auroc = {metric_dict['val_auroc'][-1]:.3f}")

    return

#create the df needed for excel, from a dict
def dict_to_df(training_dict: dict, metric_dict: dict) -> pd.DataFrame:

    final_dict = {
        'Epoch' : training_dict['Epoch'],
        'Loss' : metric_dict['train_loss'],
        'Val Loss': metric_dict['val_loss'],
        'Accuracy' : metric_dict['acc'],
        'Val Accuracy': metric_dict['val_acc'],
        'F1-Score' : metric_dict['f1_score'],
        'Val F1-Score': metric_dict['val_f1_score'],
        'Precision' : metric_dict['precision'],
        'Val Precision': metric_dict['val_precision'],
        'Recall' : metric_dict['recall'],
        'Val Recall': metric_dict['val_recall'],
        'AUROC' : metric_dict['auroc'],
        'Val AUROC': metric_dict['val_auroc'],
        'LR' : training_dict['LR'],
        'Duration' : training_dict['Duration']
    }

    training_df = pd.DataFrame(final_dict)

    return training_df

#plots the metrics using the metric_dict keys
def plot_metric(metric_dict: dict,name : str,n_epochs: int):

    train_vals = metric_dict[name]
    if name == 'train_loss': valid_vals = metric_dict['val_loss']
    else: valid_vals = metric_dict['val_'+name]
    fig,ax = plt.subplots(figsize=(12,8))
    epoch_vals = [i for i in range(1,n_epochs+1)]
    plt.plot(epoch_vals,train_vals,'bx-',epoch_vals,valid_vals,'rx-',linewidth = 4, markersize = 12)
    plt.grid(True)
    plt.title(f"{name.capitalize()} vs. Epochs",fontsize=20)
    plt.xticks(range(0,n_epochs+1))
    # plt.yticks(np.arange(0,1,0.05))
    # plt.ylabel("Loss",fontsize=14)
    plt.xlabel("Epoch",fontsize=14)
    plt.legend(["Train","Validation"])
    
    now = datetime.now()
    day = now.day
    month = now.month
    year = now.year
    plt.savefig(f'{name}_{day}_{month}_{year}.png')

    return

def metric_plots(metric_dict:dict,n_epochs:int): 

    plot_metric(metric_dict,'train_loss',n_epochs)
    plot_metric(metric_dict,'acc',n_epochs)
    plot_metric(metric_dict,'f1_score',n_epochs)
    plot_metric(metric_dict,'precision',n_epochs)
    plot_metric(metric_dict,'recall',n_epochs)
    plot_metric(metric_dict,'auroc',n_epochs)
    
    return

# does an evaluation of the model based on an unseen before test set
# calculates classification metrics
def test_model(model,loader,criterion,device):
  model.eval()

  total_loss , acc, f1_score, prec, rec, auroc = 0,0,0,0,0,0
  total_labels, total_preds, pred_probs = [],[],[]
  with torch.no_grad():
    for data in tqdm(loader,desc=f'Testing FEC model',unit='batch'):
      images,labels = data
      images,labels = images.to(device),labels.to(device)
      logits,probs = model(images)
      loss = criterion(logits,labels)
      total_loss += loss
      pred_labels = torch.argmax(probs,dim=1)
      pred_probs.append(probs)
      total_preds.append(pred_labels)
      total_labels.append(labels)

    pred_probs = torch.cat(pred_probs,dim=0)
    total_preds = torch.cat(total_preds,dim=0)
    total_labels = torch.cat(total_labels,dim=0)

    acc,f1_score,prec,rec,auroc = calculate_metrics(total_preds,total_labels,pred_probs)

  print("Test Results:")
  print('Loss: ',total_loss.item()/len(loader))
  print('Accuracy: ',acc.item())
  print('F1-Score: ',f1_score.item())
  print('Precision: ',prec.item())
  print('Recall: ',rec.item())
  print('AUROC: ',auroc.item())

  return

#plot and save the confusion matrix produced by the model evaluation on the test set
def plot_conf_matrix(model,loader,device,class_names):
  model.eval()

  total_labels, total_preds = [],[]
  with torch.no_grad():
    for data in tqdm(loader,desc=f'Testing FEC model',unit='batch'):
      images,labels = data
      images = images.to(device)
      logits,probs = model(images)
      pred_labels = torch.argmax(probs,dim=1).detach().cpu().numpy() #get most probable class
      total_labels.extend(labels)
      total_preds.extend(pred_labels)

  conf_matrix = confusion_matrix(total_labels,total_preds)
  disp = ConfusionMatrixDisplay(conf_matrix,display_labels = class_names.values())
  disp.plot(cmap = plt.cm.cividis)
  disp.im_.colorbar.remove()
  plt.xticks(rotation = 90)
  plt.grid(visible = False)
  now = datetime.now()
  day,month,year = now.day,now.month,now.year
  plt.savefig(f'conf_matrix_{day}_{month}_{year}.png',bbox_inches = 'tight')
  plt.show()

  return conf_matrix

#based on confusion matrix values, calculate TP,FP,FN and derived metrics
#Note: True Negative (TN) doesn't have much meeaning in the context of
#multiclass classification
def metrics_by_class(cf_matrix,class_names):

  num_classes = cf_matrix.shape[0]

  TP = np.diag(cf_matrix)
  FP = cf_matrix.sum(axis=0) - TP
  FN = cf_matrix.sum(axis=1) - TP

  precision = TP / (TP+FP)
  recall = TP / (TP+FN)
  f1_score = 2*(precision*recall) / (precision + recall)

  class_metrics_dict = {"Emotion": class_names.values(),"TP":TP,"FP":FP,"FN":FN,"precision":precision,"recall":recall,"f1_score":f1_score}
  class_metrics_df = pd.DataFrame(class_metrics_dict)
  print(class_metrics_df)

  return

#Essentially the equivalent of a model.fit() method in Keras,
#Pytorch offers less elegant but more customizable training and validation function
def train_model(model,train_loader,valid_loader,optimizer,scheduler,criterion,n_epochs,device,es):

    model.to(device)
    torch.backends.cudnn.benchmark = True #optimizing algorithm
    #dict that contains epoch , lr , and epoch duration values
    training_dict = {'Epoch' : [] , 'LR': [], 'Duration' : []}

    #metric names, essentially used as keys for the metric_dict
    metrics = ['train_loss','val_loss', 
               'acc','val_acc',
               'f1_score','val_f1_score',
               'precision','val_precision',
               'recall','val_recall',
               'auroc','val_auroc']
    metric_dict = {name : [] for name in metrics}

    #time variable for counting total training time of model
    model_total = timedelta()

    #total parameters of the model (in millions)
    total_parameters = sum(p.numel() for p in model.parameters())/1e6
    #print and log basic model info
    logging.info(f"\n{model.base_model_name} based {model.__class__.__name__} - Params: {total_parameters:.2f}M\n")
    print(f"\n{model.base_model_name} {model.__class__.__name__} - Params: {total_parameters:.2f}M\n")

    # epoch loop
    for epoch in range(1,n_epochs+1):
        start = time.time() #start time of training for one epoch

        init_lr = scheduler.get_last_lr()[0] # get current value of LR
        # init_lr = optimizer.param_groups[0]['lr']

        model.train() # set model to train mode, allow gradient flow
        train_loss = 0.0
        total_preds, total_labels,pred_probs = [],[],[] #lists for storing predicted labels(argmax), true labels, softmax outputs respectively
        for data in tqdm(train_loader,desc=f"Epoch {epoch}/{n_epochs} LR = {init_lr}",unit="batch",leave=True):
            images,labels = data
            images,labels = images.to(device,non_blocking=True),labels.to(device,non_blocking=True)
            for param in model.parameters():
                param.grad = None #initialize all grads to zero before backwards pass, this is faster than optimizer.zero_grad()
            logits,probs = model(images) #inference on batch of images, returns logits and softmax scores
            pred_labels = torch.argmax(probs,dim=1) #most probable class
            loss = criterion(logits,labels) #calculate loss
            loss.backward() #backpropagate the loss with gradients
            optimizer.step() #update weights
            train_loss += loss #accumulate losses
            pred_probs.append(probs)
            total_preds.append(pred_labels)
            total_labels.append(labels)

        pred_probs = torch.cat(pred_probs,dim=0)
        total_preds = torch.cat(total_preds,dim=0)
        total_labels = torch.cat(total_labels,dim=0)

        train_loss = train_loss.item()/len(train_loader) #mean loss for epoch
        acc,f1_score,prec,rec,auroc = calculate_metrics(total_preds,total_labels,pred_probs)
        append_metrics('train',metric_dict,train_loss,acc.item(),f1_score.item(),prec.item(),rec.item(),auroc.item())

        model.eval() #set model to evalution mode
        val_loss = 0
        val_preds, val_labels, val_pred_probs = [],[],[]
        with torch.no_grad(): #no gradients calculated, just validating the model performance based on training updates
            for data in tqdm(valid_loader,desc=f"Validation on Epoch {epoch}/{n_epochs}",unit="batch",leave=True):
                images,labels = data
                images,labels = images.to(device,non_blocking=True),labels.to(device,non_blocking=True)
                logits,probs = model(images)
                pred_labels = torch.argmax(probs,dim=1)
                loss = criterion(logits,labels)
                val_loss += loss
                val_pred_probs.append(probs)
                val_preds.append(pred_labels)
                val_labels.append(labels)

            val_pred_probs = torch.cat(val_pred_probs,dim=0)
            val_preds = torch.cat(val_preds,dim=0)
            val_labels = torch.cat(val_labels,dim=0)

            val_loss = val_loss.item()/len(valid_loader)
            acc,f1_score,prec,rec,auroc = calculate_metrics(val_preds,val_labels,val_pred_probs)
            append_metrics('validation',metric_dict,val_loss,acc.item(),f1_score.item(),prec.item(),rec.item(),auroc.item())

        scheduler.step(val_loss) #check if LR needs to be changed
        cur_lr = scheduler.get_last_lr()[0]
        # cur_lr = optimizer.param_groups[0]['lr']

        epoch_dur = timedelta(seconds=time.time()-start) #calculate time spent for training over 1 epoch
        model_total += epoch_dur #accumulate to total training time

        training_dict['Epoch'].append(epoch) 
        training_dict['LR'].append(cur_lr)
        training_dict['Duration'].append(str(epoch_dur)[2:7])

        log_print_info(epoch,train_loss,val_loss,cur_lr,epoch_dur,metric_dict)

        es(val_loss,model,epoch) #check if early stopping is needed
        if es.early_stop:
          es.load_best_model(model)
          break

    training_df = dict_to_df(training_dict,metric_dict)

    logging.info(f"---------------------- Total Training Time: {str(model_total)[:7]} hh:mm:ss ---------------------")
    print(f"---------------------- Total Training Time: {str(model_total)[:7]} hh:mm:ss ---------------------")

    now = datetime.now()
    day,month,year = now.day,now.month,now.year
    save_to_excel(training_df,f'training_{day}_{month}_{year}.xlsx')

    return model,metric_dict,epoch

def train_model_kfold(model,train_val_data,hyper_tune,optims,criterion,k_folds,n_epochs_per_fold,device,es,batch_size):

    torch.save(model.state_dict(),'init_model.pt')

    fold_dicts = []
    epochs_lasted_per_fold = []
    #configuration of k-fold cross validation to reduce overfitting
    kfold = KFold(n_splits=k_folds,shuffle=True)

    for fold,(train_idx,val_idx) in enumerate(kfold.split(train_val_data)):
        
        model.load_state_dict(torch.load("init_model.pt"))
        train_loader = DataLoader(dataset=train_val_data,batch_size=batch_size,
                                 sampler=torch.utils.data.SubsetRandomSampler(train_idx),num_workers=12,pin_memory=True)
        valid_loader = DataLoader(dataset=train_val_data,batch_size=batch_size,
                                 sampler=torch.utils.data.SubsetRandomSampler(val_idx),num_workers=12,pin_memory=True)
       
        # epoch loop
        logging.info(f"Fold {fold+1}")

        if hyper_tune == True:
            optimizer = optims[fold]
        else:
            optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-4)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.9,patience=2,min_lr=1e-6)
        fold_model, metric_dict, epoch = train_model(model,train_loader,valid_loader,optimizer,scheduler,criterion,n_epochs_per_fold,device,es)
        torch.save(fold_model,f"model_fold_{fold}.pt")
        fold_dicts.append(metric_dict)
        epochs_lasted_per_fold.apppend(epoch)
        
    return fold_dicts, epochs_lasted_per_fold

def save_dict(metric_dict):
    
    with open('metric_dict.json','w') as json_file:
       json.dump(metric_dict,json_file)

       print("Dictionary saved to json.")

    return