import logging
import argparse
from helpers import *
from fer_models import *

"""
Script for training model that is used in CLI in /jupyter_notebooks/train_test_cli.ipynb
"""

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Model Training.')
   parser.add_argument('--base-model',type=str,required=True,help='Choose base model, options: "efficientnet,"resnet","mobilenet","densenet","vgg"')
   parser.add_argument('--train-dir',type=str,required=True,help='Path to train set')
   parser.add_argument('--valid-dir',type=str,required=True,help='Path to validation set')
   parser.add_argument('--test-dir',type=str,required=True,help='Path to test set')
   parser.add_argument('--kfold',type=bool,required=False,help='Train with cross validation')
   parser.add_argument('--num-folds',type=int,required=False,help='Number of folds for cross validation')
   parser.add_argument('--hyper-tune',type=bool,required=False,default=False,help='Specify for hyperparameter tuning with kfold CV')
   parser.add_argument('--epochs-per-fold',type=int,required=False,help='Epochs per fold')
   parser.add_argument('--log-dir',type=str,required=False,help='Log file directory')
   parser.add_argument('--out-xlsx',type=str,required=False,help='Output dir for training excel file')
   parser.add_argument('--epochs',type=int,required=False,help='Number of training epochs')
   parser.add_argument('--batch-size',type=int,required=True,help='Image batch size for model')
   parser.add_argument('--model-save-path',type=str,required=True,help='Specify path to save model')

   args = parser.parse_args()
   base_model = args.base_model
   train_dir = args.train_dir
   val_dir = args.valid_dir
   test_dir = args.test_dir
   kfold = args.kfold
   num_folds = args.num_folds
   hyper_tune = args.hyper_tune
   epochs_per_fold = args.epochs_per_fold
   log_dir = args.log_dir
   out_xlsx = args.out_xlsx
   n_epochs = args.epochs
   batch_size = args.batch_size
   model_save_path = args.model_save_path

   train_dataset,val_dataset,test_dataset = create_dataset(train_dir,val_dir,test_dir)
   dataset_distribution(train_dataset,val_dataset,test_dataset)

   if kfold:
      train_val_dataset, test_loader = create_kfold_structs(train_dataset,val_dataset,test_dataset,batch_size)
   else:
      train_loader,val_loader,test_loader = create_data_loaders(train_dataset,val_dataset,test_dataset,batch_size)

   logging.basicConfig(
      filename = log_dir,
      level = logging.INFO,
      filemode = 'a',
      format = '%(message)s',
      force=True
      )
      
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   model = FEClassifier(base=base_model)
   es = EarlyStopping()
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.AdamW(model.parameters())
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.9,patience=2,threshold=1e-2,min_lr=1e-5)

   if kfold:
      optims = []
      fold_dicts,epochs_lasted_per_fold = train_model_kfold(model,train_val_dataset,hyper_tune,optims,scheduler,criterion,num_folds,epochs_per_fold,device,es,batch_size)
      
      save_dict(fold_dicts)
   else:
      trained_model,metric_dict,epochs_lasted = train_model(model,train_loader,val_loader,optimizer,scheduler,criterion,n_epochs,device,es)
      torch.save(trained_model.state_dict(),model_save_path)
      metric_plots(metric_dict,epochs_lasted)
      save_dict(metric_dict)

torch.save(test_loader,'test_loader.pth')