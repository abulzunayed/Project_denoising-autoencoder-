import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#from torchvision.transforms import ToTensor
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



#transform= transforms.ToTensor()    # important: this is require for Sigmoid if image size range 0 to 1
# # Define data transforms
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)) ])   ### see the last image value[-1,1] or transforms.Normalize((0.1307,), (0.3081,))
                                            # Normalize to [-1, 1] range:Normalize(std=(0.5, 0.5, 0.5), mean=(0.5, 0.5, 0.5))
                                            
# check image value . #transform= transforms.ToTensor()    # important: this is require for Sigmoid and image size 0 to 1
#transform= transforms.ToTensor()    # important: this is require for Tanh and image size -1 to 1


# prepare data loaders
mnist_train= datasets.MNIST(root='./data', train=True, download=True, transform=transform)   # Load MNIST train dataset
train_data_loader= torch.utils.data.DataLoader(dataset=mnist_train, batch_size=64, shuffle=True) # Create data loader for train

# Testing data
mnist_test= datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_data_loader= torch.utils.data.DataLoader(dataset=mnist_test, batch_size=64, shuffle=True)


#%%
##  image and noisy image visualization

dataiter=iter(train_data_loader)
images, labels=next(dataiter)
print(torch.min(images), torch.max(images))


def reshape(array):
    for images, labels in array:
        images_reshaped = images.view(images.shape[0], 1, 28, 28)  # Reshape the data to the appropriate format
        return images_reshaped

train_data = reshape(train_data_loader)
test_data = reshape(test_data_loader)

def add_noise(Y):
    noise_factor = 0.5
    noisy_train_data1 = Y + noise_factor * torch.randn(*Y.shape)
    return noisy_train_data1


# create a copy of data with added noise
noisy_train_data = add_noise(train_data)
noisy_test_data = add_noise(test_data)

def display(array1, array2):
    n = 10
    indices = np.random.randint(len(array1), size=n)
    image1 = array1[indices, :]
    image2 = array2[indices, :]

    #plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(image1, image2)):
        ax = plt.subplot(2, n, i + 1 )
        plt.imshow(image1.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# Display the train and  a version of it with added noise
display(train_data, noisy_train_data)
#%%
# Model for CNN
class Autoencoder_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # N = batch_size,  Image_size= 1,28,28
        # N,1,28,28
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # N, 16, 14, 14
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # N, 64, 1, 1
        )

        # N, 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # --> N, 32, 7,7
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # N,16, 14, 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # N,1, 28, 28
            nn.Tanh())
        # nn.Sigmoid())

    def forward(self, x):
        encoder = self.encoder(x)
        decoded = self.decoder(encoder)
        return decoded

# Note: last image_size [-1,1]-->nn.Tanh
# nn.MaxPool2d--> nn.MaxUnpool2d

model_CNN=Autoencoder_CNN()
print(model_CNN)

criterion= nn.MSELoss()
optimizer=torch.optim.Adam(model_CNN.parameters(), lr=1e-3, weight_decay=1e-5)
#optimizer=torch.optim.Adam(model_CNN.parameters(), lr=0.1)
#%%
# train the Model
train_losses = []
num_epochs = 50
outputs_CNN = []

# for adding noise to images
#noise_factor = 0.4

for epoch in range(num_epochs):
    # for (img, _) in train_data_loader:
    # for i, (images, labels) in enumerate(train_loader):

    for data in train_data_loader:
        images, _ = data

        ## add random noise to the input images
 #       noisy_imgs = images + noise_factor * torch.randn(*images.shape)
        # Clip the images to be between 0 and 1
 #       noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        noisy_imgs=add_noise(images)
        recon_CNN = model_CNN(noisy_imgs)

        loss_CNN= criterion(recon_CNN, images)

        optimizer.zero_grad()
        loss_CNN.backward()
        optimizer.step()

        train_losses.append(loss_CNN.item())
        recon_image = recon_CNN.view(images.shape[0], 1, 28, 28)
        outputs_CNN.append((epoch, noisy_imgs, recon_image))

        print(f'Epoch: {epoch + 1}, Loss:{loss_CNN.item():.4f}')
#%%
#Save model_CNN Model

torch.save(model_CNN.state_dict(), 'U:\Dream\Autoencoder\model_CNN.pth')

model_CNN.load_state_dict(torch.load('U:\Dream\Autoencoder\model_CNN.pth'))
model_CNN.eval()

#%%
#Plot loss values

#plt.figure(figsize=(10,5))
plt.title("Loss vs epoch for training")
plt.plot(train_losses,label="loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

for k in range(0, num_epochs,5):
    #plt.figure(figsize=(20, 6))
    plt.gray()
    #imgs = outputs_CNN[k][1].detach().numpy()  # convert torch to numpy
    noise_img = outputs_CNN[k][1].detach().numpy()
    recon = outputs_CNN[k][2].detach().numpy()

    display(noise_img, recon)
    
    

    
#%%
#  test image : obtain one batch of test images

# Load model_CNN Model
model_CNN.load_state_dict(torch.load('U:\Dream\Autoencoder\model_CNN.pth'))

dataiter = iter(test_data_loader)
images, labels = next(dataiter)
print(torch.min(images), torch.max(images))

# add noise by calling function
noisy_imgs=add_noise(images)
# get sample outputs
output = model_CNN(noisy_imgs)       # train test image
# prep images for display
noisy_imgs = noisy_imgs.numpy()

# output is resized into a batch of iages
output = output.view(images.shape[0], 1, 28, 28)
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

# input images on top row, reconstructions on bottom
for noisy_imgs, row in zip([noisy_imgs, output], axes):
    for img, ax in zip(noisy_imgs, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
#%%

#Accuracy
     
# =============================================================================
# =============================================================================
# from tqdm import tqdm
# 
# def test():
#     print("Model testing ....")
#     model_CNN.load_state_dict(torch.load('U:\Dream\Autoencoder\model_CNN.pth'))  #load model
#     model_CNN.eval()
#     test_data_loader= torch.utils.data.DataLoader(dataset=mnist_test, batch_size=64, shuffle=False)
#     predictions, targets = [], []
#     
#     correct, total = 0, 0
#     for images, labels in tqdm(test_data_loader, leave=False):
#         pred = model_CNN(images)
#         print(labels.shape)
#         
#         _, predicted = torch.max(pred.data, 1)

#pred_y = torch.max(test_output, 1)[1].data.squeeze()
#test_accuracy = (sum(pred_y == test_y).item() / test_y.size(0)) * 100

#         print(predicted.shape)
#         predictions.extend(predicted.numpy().tolist())
#         targets.extend(labels.numpy().tolist())
#         
#         # calculate accuracy
#         total += labels.size(0)
#         correct += (predicted== labels).sum().item()
#     print(f"Accuracy: {correct / len(mnist_test) * 100} %",)
#     
# # call the accuracy for the model
# test()
# =============================================================================
# =============================================================================
        
#%%
# Make a Linear Model
# =============================================================================
# 
# class Autoencoder_Linear(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #N = batch_size, 784= Image_size
#         self.encoder= nn.Sequential(
#             nn.Linear(28*28, 128),    # N, 784 -->N, 128
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 12),
#             nn.ReLU(),
#             nn.Linear(12, 3)           # --> N, 3
#             )      
#         
#         self.decoder= nn.Sequential(
#             nn.Linear(3, 12),    # --> N, 
#             nn.ReLU(),
#             nn.Linear(12, 64),
#             nn.ReLU(),
#             nn.Linear(64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 28*28),   # --> N, 784
#            # nn.Sigmoid())
#             nn.Tanh())
#         
#     def forward(self, x):
#         encoder=self.encoder(x)
#         decoded =self.decoder(encoder)
#         return decoded
#     
# model=Autoencoder_Linear()
# print(model)
# 
# criterion= nn.MSELoss()
# optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# 
# # Train Autoencoder_ Linear
# train_losses=[]
# num_epochs= 6
# outputs_linear=[]
# noise_factor=0.4
# 
# 
# for epoch in range(num_epochs):
#     for data in train_data_loader :
#         # img= torch.Size([64, 1, 28, 28])
#         images,_ =data
#         img1=images.reshape(-1, 28*28) 
#         
#         ## add random noise to the input images
#         noisy_imgs = images + noise_factor * torch.randn(*images.shape)
#         # Clip the images to be between 0 and 1
#         noisy_imgs = np.clip(noisy_imgs, 0., 1.)
#         
#         noise=noisy_imgs.reshape(-1, 28*28)      #img1=torch.Size([64, 784])
#                
#         recon_linear= model(noise)              
#         
#         loss=criterion(recon_linear, img1)
# 
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
# 
#         train_losses.append(loss.item())
#         #noisy_image = noisy_imgs.view(img.shape[0], 1, 28, 28)
#         recon_image = recon_linear.view(images.shape[0], 1, 28, 28)
#         
#         outputs_linear.append((epoch, images, noisy_imgs, recon_image))
#         
#     print(f'Epoch: {epoch+1}, Loss:{loss.item():.4f}')
#            
#     
# 
# plt.figure(figsize=(10,5))
# 
# plt.plot(train_losses,label="loss")
# 
# 
# plt.title("Loss vs epoch")
# plt.xlabel("num_epochs")
# plt.ylabel("train_losses")
# plt.legend()
# plt.show()
# 
# for k in range(0, num_epochs,1):
#     #plt.figure(figsize=(20, 6))
#     plt.gray()
#     #imgs = outputs_CNN[k][1].detach().numpy()  # convert torch to numpy
#     noise_img = outputs_linear[k][2].detach().numpy()
#     recon = outputs_linear[k][3].detach().numpy()
# 
#     display(noise_img, recon)
# 
# =============================================================================
