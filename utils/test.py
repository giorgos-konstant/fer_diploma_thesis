import torch.nn as nn
import argparse
from helpers import *
from fer_models import *

"""
Script for testing model that is used in CLI in /jupyter_notebooks/train_test_cli.ipynb
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training.')
    parser.add_argument('--base-model',type=str,required=True,help='Choose base model, options: "efficientnet,"resnet","mobilenet","densenet"')
    parser.add_argument('--model-path',type=str,required=True,help='Path to stored model')
    parser.add_argument('--test-loader-path',type=str,required=True,help='Path to test DataLoader')
    
    args = parser.parse_args()
    base_model = args.base_model
    model_path = args.model_path
    test_loader_path = args.test_loader_path


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    test_loader = torch.load(test_loader_path)
    model = FEClassifier(base=base_model)
    model.load_state_dict(torch.load(model_path))
    model.to('cuda')
    class_names = {0: 'anger',1:'disgust',2:'fear',3:'happy',4: 'neutral', 5:'sad',6:'surprise'}

    test_model(model,test_loader,criterion,device)
    conf_matrix = plot_conf_matrix(model,test_loader,device,class_names)
    metrics_by_class(conf_matrix,class_names)