# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 11:14:36 2023

@author: zunayeal
"""

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import normalize

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#%% data Preparation
D=loadmat(r"Example_fit.mat")

Xz = torch.Tensor(D['Z_corrExt'])

plt.figure(1)
X_image= Xz[:,:,0,7]
inp=plt.imshow(X_image, cmap='gray', vmin=0.05, vmax=0.25)
plt.colorbar(inp)
plt.title(label='input Image: Z_corrExt[:,:,0,7]')
plt.show()

X1 = torch.reshape(Xz, (Xz.shape[0]*Xz.shape[1]*Xz.shape[2], Xz.shape[3]))    # rehsape to 2D and shape=torch.Size([13824, 57])
XX = X1.T                                         # transpose X,shape=torch.Size([57, 13824])

Yp = torch.Tensor(D['popt'])

plt.figure(2)
Y_image= Yp[:,:,0,7]
tar=plt.imshow(Y_image, cmap='gray', vmin=0.05, vmax=0.25)
plt.colorbar(tar)
plt.title(label='Target Image: popt[:,:,0,7]')
plt.show()

Y1 = torch.reshape(Yp, (Yp.shape[0]*Yp.shape[1]*Yp.shape[2], Yp.shape[3]))    # rehsape to 2D and shape=torch.Size([13824, 16])
YY = Y1.T                                          # transpose X,shape=torch.Size([16, 13824])

mask1 = torch.isnan(XX)                           # mask1.shape= torch.Size([57, 13824])
mask2 = torch.isnan(YY)                           # mask1.shape=torch.Size([16, 13824])

# =============================================================================
#  mask_index_1 is a binary mask tensor with same number of columns as mask1. It is initialized to all False values, 
#  and later will be updated in the loop to indicate which columns of mask1 have at least one True value.
# 
# =============================================================================
mask_index_1 = torch.zeros([1, mask1.shape[1]])               # torch.Size([1, 13824])
mask_index_2 = torch.zeros([1, mask2.shape[1]])               #torch.Size([1, 13824])

""" Loop for only 1st row and all cols"""
for col in range(mask1.shape[1]):                             
    if True in mask1[:, col]:
        mask_index_1[0, col] = True        # If there is 'true' value, it sets the corresponding element of mask_index_1 to True.

for col in range(mask2.shape[1]):                              
    if True in mask2[:, col]:
        mask_index_2[0, col] = True        

# =============================================================================
# 'nonzero' function to get the indices where the new tensor is 'True'. The 'as_tuple=False' argument tells nonzero to return
#  a tensor of shape (num_indices, num_dims),where num_indices is number of indices where the mask is True, and num_dims is 
#  the number of dimensions. Finally all in NAN value indices converts to 1, othewise 0. 
# =============================================================================

mask_index_1 = (mask_index_1 == 1).nonzero(as_tuple=False)     
mask_index_2 = (mask_index_2 == 1).nonzero(as_tuple=False)     

mask_final = torch.cat((mask_index_1, mask_index_2)).unique()    #  torch.Size([7732])

# create new tensor without NAN values by subtracting NAN value indices.

X_ohne_Nan = torch.zeros([XX.shape[0], XX.shape[1] - mask_final.shape[0]])    # torch.Size([57, 6092]) --> Zeros tensor
Y_ohne_Nan = torch.zeros([YY.shape[0], YY.shape[1] - mask_final.shape[0]])

# without NAN value for 57 rows
X_ohne_Nan = XX[:, [i for i in range(XX.shape[1]) if i not in mask_final]]    
Y_ohne_Nan = YY[:, [i for i in range(YY.shape[1]) if i not in mask_final]]   

print(X_ohne_Nan.shape)   # torch.Size([57, 6092])
print(Y_ohne_Nan.shape)   # torch.Size([16, 6092])

#%% Transpose to the data and Final input
X=X_ohne_Nan.T   # torch.Size([6092, 57])
T=Y_ohne_Nan.T   # torch.Size([6092, 16])
print(X.shape)   
print(T.shape)

#%%


#%%   Create model
hiddenLayerSize = [100, 200, 100]
class Deep_net(nn.Module):
     def __init__(self):
         super().__init__()
         
         self.encoder= nn.Sequential(nn.Linear(X.shape[1], hiddenLayerSize[0]),              #57, 100)
                         nn.ReLU(), nn.Linear(hiddenLayerSize[0], hiddenLayerSize[1]),    #(100, 200)
                         nn.ReLU(), nn.Linear(hiddenLayerSize[1], hiddenLayerSize[2]),     #(200, 100)
                         nn.ReLU(), nn.Linear(hiddenLayerSize[2], T.shape[1]))             # (100, 16)
                     #nn.Sigmoid())      
     
     
     def forward(self, x):
         encoder=self.encoder(x)
         return encoder
#%%  print model 
model_net=Deep_net()
print(model_net)
 
 #%% optimizer
 
criterion= nn.MSELoss()
optimizer=torch.optim.Adam(model_net.parameters(), lr=1e-3, weight_decay=1e-5)
 #%%  split data

from sklearn.model_selection import train_test_split
(trainX, testX, trainY, testY) = train_test_split(X, T,test_size=0.15, random_state=95) 

print(trainX.shape)  # torch.Size([5178, 57])
print(trainY.shape)  # torch.Size([5178, 16])
print(testX.shape)   # torch.Size([914, 57])
print(testY.shape)   # torch.Size([914, 16])
print(trainX.shape[0])  # 5178
#%% plot a sample data

plt.figure(3)
plt.plot(trainX[5,:])


#%% define  batch function

def next_batch(inputs, targets, batchSize):
    # loop over the dataset                                                                    
    for i in range(0, inputs.shape[0], batchSize):             # print(inputs.shape[0])= 5178 and gradually "i" will be increase by 64
        # yield a tuple of the current batched data and labels
        yield (inputs[i:i + batchSize], targets[i:i + batchSize])    # each batch size of "inputs[i:i + batchSize]"=torch.Size([64, 57])
        # yield statement produces a generator object and can return multiple values to the caller without terminating the program, 
            
#%% define batch size and epoch

BATCH_SIZE = 64
EPOCHS = 40


#%%  training the model

train_losses = []
    # loop through the epochs
for epoch in range(0, EPOCHS):	
    print("[INFO] epoch: {}...".format(epoch + 1))
    trainLoss = 0
    #trainAcc = 0
    samples = 0
    model_net.train()

    
    for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):    # per epoch 5178/64= 81 loop
        # print(batchX.shape) # torch.Size([64, 57])   
        predictions = model_net(batchX) 
        
        loss = criterion(predictions, batchY.float())   #         prediction = torch.Size([64, 16]) and batchY = torch.Size([64, 16]) 
        
        # zero the gradients accumulated from the previous steps,
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # update training loss, accuracy, and the number of samples
        trainLoss += loss.item() * batchY.size(0)            # batchY.size(0)=64
        samples += batchY.size(0)
        #trainAcc += (predictions.max(1)[1] == batchY).sum().item()
        
    # display model progress on the current training batch
    trainTemplate = "epoch: {} train loss: {:.3f}"
    print(trainTemplate.format(epoch + 1, (trainLoss / samples)))
    
    #trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
    #print(trainTemplate.format(epoch + 1, (trainLoss / samples),(trainAcc / samples)))
    
    train_losses.append((trainLoss/ samples))
    
#%% plot train loss 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

plt.figure(4)
plt.title("Traing Loss ")
plt.plot(train_losses,label="trainloss")

plt.xlabel("Epoch")
plt.ylabel("trainLoss")
plt.legend()
plt.show()

#%% save model
#torch.save(model_net.state_dict(), 'U:\Dream\Autoencoder\E500_V7_Deep_net.pth')
torch.save(model_net.state_dict(), 'U:\Dream\Autoencoder\E40_V7_Deep_net.pth')

#%% Restore the model for testing
# =============================================================================
model_net_test=Deep_net()
print(model_net_test)
#model_net_test.load_state_dict(torch.load('U:\Dream\Autoencoder\E500_V7_Deep_net.pth'))
model_net_test.load_state_dict(torch.load('U:\Dream\Autoencoder\E40_V7_Deep_net.pth'))

# #%%  model for testing data
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import r2_score
# 
# test_losses = [] 
# testLoss = 0
# samples = 0
# mse_error = 0
# #r2_score=0
# MSE_error = [] 
# R2_score = [] 
# model_net_test.eval()
# # initialize a no-gradient context
# with torch.no_grad():
#     for (batchX, batchY) in next_batch(testX, testY, BATCH_SIZE):   # 914/64= 15 loop
#         # print(batchX.shape) # torch.Size([64, 57])   
#         predictions = model_net_test(batchX) 
#         
#         loss = criterion(predictions, batchY.float())           # prediction = torch.Size([64, 16]) and batchY = torch.Size([64, 16]) 
#                                                                     # Here testloss and MSE error should be same value
#         # MSE Error nad R2 score
#         err = mean_squared_error(predictions, batchY.float())        
#         r2 = r2_score(batchY.float(),predictions, multioutput='variance_weighted')
#         # update training loss, accuracy, and the number of samples
#         testLoss += loss.item() * batchY.size(0)            # batchY.size(0)
#         samples += batchY.size(0)
#         
#         mse_error += err.item() * batchY.size(0)
#         #R2_score += r2.item() * batchY.size(0)
#         #print(R2_score)
#       # display model progress on the current training batch
#       
#         testTemplate = " test loss: {:.3f}  MSE_error: {:.3f} R2_score: {:.3f}"
#         print(testTemplate.format( (testLoss / samples),(mse_error/ samples) , ( r2)))
#         test_losses.append((testLoss/ samples))
#         MSE_error.append((testLoss/ samples))
#         R2_score.append((r2))
#         
# r=sum(R2_score) / len(R2_score)
# print(f"R2_score: {r}")
# #%% plot for testing data
# plt.figure(5)
# plt.title("Testing Loss ")
# plt.subplot(1,3,1)
# plt.plot(test_losses,label="testloss")
# plt.xlabel("Epoch")
# plt.ylabel("testLoss")
# plt.legend()
# plt.show()
# 
# plt.subplot(1,3,2)
# plt.plot(MSE_error,label="MSE_error")
# plt.xlabel("Epoch")
# plt.ylabel("MSE_error")
# plt.legend()
# plt.show()
# 
# plt.subplot(1,3,3)
# plt.plot(R2_score,label="R2_score")   
# plt.xlabel("Epoch")
# plt.ylabel("R2_score")
# plt.legend()
# plt.show()
# 
# =============================================================================
#%%
# reconstruction image from prediction

model_net_Con=Deep_net()
print(model_net_Con)
#model_net_Con.load_state_dict(torch.load('U:\Dream\Autoencoder\E500_V7_Deep_net.pth'))
model_net_Con.load_state_dict(torch.load('U:\Dream\Autoencoder\E40_V7_Deep_net.pth'))

# X=X_ohne_Nan.T   # torch.Size([6092, 57])   # without NAN
# T=Y_ohne_Nan.T   # torch.Size([6092, 16])    # withot NAN target

count=0
Pred_original = torch.zeros([YY.shape[0], YY.shape[1]])     # (16, 13824)

'''
Pass input image thorugh Model
'''
Pred = model_net_test(X)   # model input X= (6092, 57) and output, pred=(6092, 16)
Pred=Pred.T   # Pred= (16, 6092)

# Need to convert mask_final tensor to list
mask_final_list = mask_final.tolist()     # list.size= 7732

for i in range(Pred_original.shape[1]):
    try:
        if mask_final_list.index(i) != -1:  # Means NAN values in this column
            Pred_original[:, i] = np.NaN

    except Exception as e:
        Pred_original[:, i] = Pred[:, count]    #Pred_original=(16, 13824) and Pred= (16, 6092)
        count += 1
        
# final Pred_original =(16,13824)
Pred_original =Pred_original.T   #Pred_original =(13824, 16)
Y1 = torch.reshape(Yp, (Yp.shape[0]*Yp.shape[1]*Yp.shape[2], Yp.shape[3]))
recon_image=torch.reshape(Pred_original, (Yp.shape[0], Yp.shape[1], Yp.shape[2],Yp.shape[3]))
print(recon_image.shape)


reco_image1=recon_image.detach().numpy()

plt.figure(6)
reco_image2= reco_image1[:,:,0,7]

im=plt.imshow(reco_image2, cmap='gray', vmin=0, vmax=1)
plt.colorbar(im)
plt.title(label='Prection image from Model:[:,:,0,7]')
plt.show()


#%%

# predection image --> Pred_original =Pred_original.T   #Pred_original =(13824, 16)
predection_image =Pred_original.T                          # #Pred_original =(16,13824)
print(predection_image.shape)
print(YY.shape)                                  # Y,shape=torch.Size([16, 13824])

dev_img=torch.abs(YY - predection_image)

dev_img=dev_img.T
sub_image=torch.reshape(dev_img, (Yp.shape[0], Yp.shape[1], Yp.shape[2],Yp.shape[3]))


plt.figure(7)
sub1_image1=sub_image.detach().numpy()
sub_image2= sub1_image1[:,:,0,7]
im=plt.imshow(sub_image2, cmap='viridis', vmin=0.05, vmax=0.25)
plt.colorbar(im)
plt.title(label='Subtract image from Model:[:,:,0,7]')
plt.show() 

#%%
 # different type of color map
# =============================================================================
# plt.figure(7)
# reco_image2= reco_image1[:,:,0,7]
# #plt.imshow(reco_image2, cmap='gray')    # --> only gray image
# 
# #import matplotlib.image as mpimg
# plt.imshow(reco_image2, cmap='viridis')    # ----> for virdis image
# 
# #im=plt.imshow(reco_image2, cmap = plt.get_cmap('jet'))    # --> jet color image
# # im=plt.imshow(reco_image2, cmap='gray', vmin=0, vmax=1)
# plt.colorbar(im)
# plt.title(label='Prection image from Model:[:,:,0,7]')
# plt.show()
# =============================================================================
