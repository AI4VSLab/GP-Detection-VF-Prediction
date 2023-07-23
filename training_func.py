import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from Model import GPTS
from Dataloader import dataloaders
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")      # set device to GPU 0 if available, otherwise use CPU
model = GPTS().to(device)                                                    # move model to GPU
optimizer = SGD(model.parameters(),lr=0.1)                                   # define optimizer

EPOCHS = 5

train_epoch_loss = []                    # training loss per epoch
validation_epoch_loss = []               # validation loss per epoch

train_accuracy = []                      # training accuracy per epoch
validation_accuracy = []                 # validation accuracy per epoch

criterion = nn.BCEWithLogitsLoss()       # use to compute loss

for epoch in range(EPOCHS):
    
    for phase in ['train', 'validation', 'test']:
        temp_loss = []                   # stores loss at end of each iteration, dummy list
        predict_labels = []              # stores binary predictions
        true_labels = []                 # stores true labels representing correct classification for each input sample
    
        if phase=='train':
            model.train()
        else:
            model.eval()
        
        # TRAINING / VALIDATION LOOP
        for index, (data, label) in enumerate(dataloaders[phase]):
            data = data.to(device)                                  # move to GPU
            label = label.unsqueeze(1).to(device)                   # unsqueeze adds new dimension to tensor
            
            label_predicted = model(data.float())                   # OUTPUT, convert data to float so continuous
            
            loss = criterion(label_predicted, label.float())        # compute loss w/ criterion, convert label to float so continuous
            temp_loss.append(loss.data.item())
            
            if phase=='train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            preds_binary = (label_predicted>=0)                     # creates binary predictions with 0 as threshold, true(>0), false(<0)
            preds_binary_array = preds_binary.cpu().numpy()
            labels_array = label.data.cpu().numpy()
            
            predict_labels.extend(preds_binary_array)
            true_labels.extend(labels_array)
        
        accuracy = accuracy_score(true_labels, predict_labels)      # compute accuracy for epoch
        
        if phase=='train':
            train_epoch_loss.append(np.mean(temp_loss))
            train_accuracy.append(accuracy)
        if phase=='validation':
            validation_epoch_loss.append(np.mean(temp_loss))
            validation_accuracy.append(accuracy)

    torch.save(model.state_dict(), './saved_models_GPTS/checkpoint_epoch_%s.pth' % (epoch))

    print("Epoch: {} | train_loss: {} | validation_loss: {}".format(epoch, train_epoch_loss[-1], validation_epoch_loss[-1]))

train_accuracy_avg = (sum(train_accuracy)/len(train_accuracy))*100                      # avg training accuracy across all epochs
validation_accuracy_avg = (sum(validation_accuracy)/len(validation_accuracy))*100       # avg validation accuracy across all epochs

print("Training accuracy average:", round(train_accuracy_avg, 2), "%")
print("Validation accuracy average:", round(validation_accuracy_avg, 2), "%")
print("Training and validation complete!")