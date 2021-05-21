import os
import cv2
import numpy as np
from tqdm import tqdm
# from pyAudioAnalysis import MidTermFeatures as aFm
# from pyAudioAnalysis import audioBasicIO as aIO
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
# import random
import shutil

import warnings
warnings.filterwarnings('ignore')



REBUILD_DATA = False

class SirenVSChildren():
    IMG_SIZE = 128
    SIRENS = "Siren"
    CHILDREN = "Children"
    LABELS = {SIRENS : 0, CHILDREN : 1}
    training_data = []
    sirencount = 0
    childrencount = 0
    
    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                # try:
                path = os.path.join(label,f)
                y, sr = librosa.load(path, duration=2.97)
                ps = librosa.feature.melspectrogram(y=y, sr=sr)
                ps = cv2.resize(ps, (self.IMG_SIZE, self.IMG_SIZE))
                self.training_data.append([np.array(ps), np.eye(2)[self.LABELS[label]]])
            
                if label == self.SIRENS:
                    self.sirencount += 1
            
                elif label == self.CHILDREN:
                    self.childrencount += 1
                # except Exception as e:
                #     pass
                
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("nb 1 =",self.sirencount)
        print("nb 2 =",self.childrencount)
        
        
        
if REBUILD_DATA:
    data = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
    for i in range(len(data)):
        if data['classID'][i] == 8:
            dt = str(data['slice_file_name'][i])
            path = 'UrbanSound8K/audio/fold' + str(data['fold'][i]) + '/' + dt
            new_path = 'Siren/' + dt
            shutil.move(path, new_path)
        elif data['classID'][i] == 2:
            dt = str(data['slice_file_name'][i])
            path = 'UrbanSound8K/audio/fold' + str(data['fold'][i]) + '/' + dt
            new_path = 'Children/' + dt
            shutil.move(path, new_path)
    sirenvchildren = SirenVSChildren()
    sirenvchildren.make_training_data()
    
training_data = np.load("training_data.npy", allow_pickle=True)

# training_data_bis = [[0]*136,[0,0]]*20
# for i in range(len(training_data)):
#     for j in range(len(training_data[0])):
#         training_data_bis[i][j] = training_data[i][j][0]



import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, 5)
        self.conv2 = nn.Conv2d(24, 48, 5)        
        self.conv3 = nn.Conv2d(48, 48, 5)

        self.fc1 = nn.Linear(2400, 64)
        self.fc2 = nn.Linear(64, 2)        
        

    
    def forward(self,x):

        x = F.max_pool2d(F.relu(self.conv1(x)), (4,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (4,2))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1,2400)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)



net = Net()

import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr = 0.01)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 128, 128)
y = torch.Tensor([i[1] for i in training_data])


VAL_PCT = 0.2
val_size = int(len(X)*VAL_PCT)
 
train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]



BATCH_SIZE = 100
EPOCHS = 1

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
        batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,128,128)
        batch_y = train_y[i:i+BATCH_SIZE]
        
        net.zero_grad()
        outputs = net(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
print(loss)

correct = 0
total = 0
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_X[i].view(-1,1,128,128))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct +=1
        total +=1
        
print("Accuracy :", round(correct/total,3))
        
        

















        
