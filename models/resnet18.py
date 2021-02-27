# Library imports
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim 
import numpy as np 
import pandas as pd
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt 
import time 
import os 
import copy 
from tqdm import tqdm

# Define input feature map size
INPUT_SIZE = 224

class ResNet18:
    def __init__(self, input_data_dir = "./data",
                 output_progress_dir = "./output/resnet18/progress",
                 output_model_dir = "./output/resnet18/models",
                 on_split,
                 batch_size = 50, 
                 num_epochs = 50,
                 num_classes = 2,
                 training = True,
                 ):
        '''
        The ResNet18 class gets the data directory as input, as well as the hyperparameters for
        training, and outputs the training progress and the trained model as a .pth file.
        '''
        # Initialize global variables
        self.data_dir = input_data_dir
        self.output_dir = output_progress_dir
        self.output_model_dir = output_model_dir+"split_"+str(on_split)+"/"
        self.training = training
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.input_size = INPUT_SIZE
        
        # Whether to use pretrained weights from ImageNet or not
        feature_extract = True

    def set_parameter_requires_grad(self, model, feature_extracting):
        '''
        Receives the model and a the boolean feature_extracting variable, which if 
        set to True, uses the pretrained weights from ImageNet and updates the parameters of
        the model accordingly.
        '''
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, num_classes, feature_extract, use_pretrained=True):
        '''
        Receives the output number of classes, feature_extract, and the use_pretrained flag
        and outputs the initialized model.
        '''
        # Initialize input size and model variable
        model_ft = None
        input_size = 0
        # Initialize model        
        model_ft = models.resnet18(pretrained = use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        return model_ft

    def get_dataloaders(self, data_dir, bs, input_size):
        '''
        Takes the data directory as input, the batch size and the input size, and outputs
        the torch dataloader containing the transformed images in batch of bs size.
        '''
        # Define the data transformations
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        # Create training and validation datasets
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

        # Create training and validation dataloaders
        dataloaders_dict[str(split)] = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs, shuffle=True, num_workers=4) for x in ['train', 'val']}

        return dataloaders_dict

    def train_model(self, model, dataloaders, criterion, optimizer, device, num_epochs=25):
        '''
        Takes a pretrained model, the dataloaders, the momentum criterium, the optimizer, the torch device, the number
        of epochs and outputs the trained model with the progress information.
        '''
        # Start counting the training time
        since = time.time()

        # Initialize variables that will store the training information
        train_acc_history = []
        train_loss_history = []
        val_acc_history = []
        val_loss_history = []

        # Get the best model weights from the inputted model
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        
        # Loop through epochs to start training
        for epoch in tqdm(range(num_epochs)):
            print('Epoch {}/{}'.format(epoch, num_epochs -1))
            print('-'*10)

            # Each epoch has a training and validation pass
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()       # Set model to training mode
                else:
                    model.eval()        # Set model to evaluate mode
                
                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                print("Starting training ...")
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # Backpropagation + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val':
                    val_acc_history.append(epoch_acc.numpy())
                    val_loss_history.append(epoch_loss)
                else:
                    train_acc_history.append(epoch_acc.numpy())
                    train_loss_history.append(epoch_loss)

        # Calculate total training time
        time_elapsed = time.time() - since
        print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model, val_acc_history, val_loss_history, train_acc_history, train_loss_history, best_acc.numpy(), time_elapsed

    def run_resnet18(self, num_epochs=15, lr=0.01, momentum=0.9):
        print("Initializing Datasets and Dataloaders...")
        # Get cross-validation data
        loader = get_dataloaders(self.data_dir, self.batch_size, self.input_size)
        
        # Detect if we have a GPU available
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize the model for this run
        model_ft = initialize_model(self.num_classes, self.feature_extract, use_pretrained=True)

        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Initialize optimizer
        params_to_update = model_ft.parameters()

        if feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)

        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    pass

        # Define optimizer
        optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=0.9)

        # Setup the loss function
        criterion =  nn.CrossEntropyLoss()

        # Train model
        model_ft, val_acc_history, val_loss_history, train_acc_history, train_loss_history, best_acc, time_elapsed = train_model(
            model_ft, loader, criterion, optimizer_ft, device, num_epochs=num_epochs)
        
        # Record data
        training_output = {}
        training_output['train_loss'] = train_loss_history
        training_output['train_acc'] = train_acc_history
        training_output['val_loss'] = val_loss_history
        training_output['val_acc'] = val_acc_history
        training_output['best_acc'] = [best_acc]*num_epochs
        training_output['runtime(s)'] = [time_elapsed]*num_epochs
        
        # Output data
        root_output_dir = self.output_model_dir
        df_name = root_output_dir+"progress/train_progress.csv"
        print(f"Outputting data to {df_name}...")
        pd.DataFrame.from_dict(training_output).to_csv(df_name, index=False)
        output_w_dir = root_output_dir+'weights/trained_model.pth'
        torch.save(model_ft.state_dict(), output_w_dir)
        print(f"Saving final trained model to {output_w_dir}")


#----------------------------------------------------------------------------------------------
#                                       MAIN FUNCTION                                             
#----------------------------------------------------------------------------------------------

if __name__ == "__main__":

 