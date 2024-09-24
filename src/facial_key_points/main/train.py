import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from src.facial_key_points.config.config import configuration
from src.facial_key_points.datasets.datasets import FaceKeyPointData
from src.facial_key_points.model.modified_vgg import get_model
from src.facial_key_points.utils.utils import train, visualization, plot_curve
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json


def main():
    
    saved_path = os.path.join(os.getcwd(), 'dump', configuration.get('saved_path'))
    model_path = os.path.join(saved_path, 'model.pth')
    hyperparameter_path = os.path.join(saved_path, 'hyperparameter.json')
    train_curve_path = os.path.join(saved_path, 'train_curve.png')
    vis_result_path = os.path.join(saved_path, 'vis_result.png')

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    training_data = FaceKeyPointData(csv_path=configuration.get('train_data_csv_path'), split='training', device=device)
    test_data = FaceKeyPointData(csv_path=configuration.get('test_data_csv_path'), split='test', device=device)

    train_dataloader = DataLoader(training_data, batch_size=configuration.get('batch_size'), shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=configuration.get('batch_size'), shuffle=True)

    model = get_model(device=device)

    criterion = nn.L1Loss()
    optimizer= torch.optim.Adam(model.parameters(), lr=configuration.get('learning_rate'))

    train_loss, test_loss = train(configuration.get('n_epochs'), train_dataloader, test_dataloader, model, criterion, optimizer)
    # print( configuration.get('model_input_size'))
    # plot_curve(train_loss, test_loss, train_curve_path)
    visualization('face.jpg', model, vis_result_path, configuration.get('model_input_size'), device)

    with open(hyperparameter_path,'w') as h_fp:
        json.dump(configuration, h_fp)

    torch.save(model, model_path)





if __name__ == "__main__":
    main()