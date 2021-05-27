import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from InceptionNet import *
import sys
# Useful for examining the network
from torchsummary import summary


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')



@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    val_losses = []
    val_acc = []

    for images, labels in tqdm(val_loader):
        
        images = images.to(device=device)
        labels = labels.to(device=device)

        out = model(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        val_losses.append(loss)
        val_acc.append(acc)

    epoch_loss = torch.stack(val_losses).mean()
    epoch_acc  = torch.stack(val_acc).mean()

    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


def fit(epochs, lr, model, train_loader, val_loader, device, opt_func=optim.SGD, enable_aux=False):
    history = []
    optimizer = opt_func(model.parameters(), lr=lr)
    
    for epoch in range(epochs):

        print(f"Training --- Epoch:- {epoch+1}/{epochs}")

        # Training Phase 
        model.train()
        train_losses = []
        for images, labels in tqdm(train_loader):
            
            images = images.to(device=device)
            labels = labels.to(device=device)

            if enable_aux:
                aux1_out, aux2_out, main_out = model(images)
                aux1_loss = F.cross_entropy(aux1_out, labels)
                aux2_loss = F.cross_entropy(aux2_out, labels)
                main_loss = F.cross_entropy(main_out, labels)

                loss = main_loss + 0.3*(aux1_loss+aux2_loss)

            else:
                out = model(images)
                loss = F.cross_entropy(out, labels)

            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            
        # Validation phase
        result = evaluate(model, val_loader, device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        history.append(result)

        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))

    return history



device = get_default_device()

print("[LOG] ---- Training Device: ", device)

image_transform = {
	'train': transforms.Compose([transforms.CenterCrop(size=224), transforms.ToTensor()]),
	'test' : transforms.Compose([transforms.CenterCrop(size=224), transforms.ToTensor()])
}

# Directory of the trainin and test data
data_dir = 'dataset'

train_dataset = ImageFolder(data_dir+'/train_set', transform=image_transform['train'])
test_dataset = ImageFolder(data_dir+'/test_set', transform=image_transform['test'])

img, label = train_dataset[0]
print("[LOG] ---- Shape of input Image: ", img.shape, "True Label: ", label)

torch.manual_seed(20)
val_size = 1000
train_size = len(train_dataset) - val_size

train_ds, val_ds = random_split(train_dataset, [train_size, val_size])
print("[LOG] ---- Number of training images: ", len(train_ds))
print("[LOG] ---- Number of validation images: ", len(val_ds))

batch_size = 8
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Debug - Purposes
# for images, labels in tqdm(train_loader):
#     print(images.shape)
#     print(labels.shape)
#     print(labels)
#     break

num_epochs = 10
opt_func = optim.Adam
lr = 0.01

# To include auxillary classifiers for training - default = False, if not provided
ENABLE_AUX = True

# To include batchnorm layers for training - default = True, if not provided
USE_BATCHNORM = True

model = GoogLeNet(num_classes=2, aux_network=ENABLE_AUX, use_batchnorm=USE_BATCHNORM).to(device=device)

history = fit(num_epochs, lr, model, train_loader, val_loader, device, opt_func=opt_func, enable_aux=ENABLE_AUX)

print("[LOG] ---- Evaluating on Test Set")
test_result = evaluate(model, test_loader, device)
print(f"Test Loss: {test_result['val_loss']:.4f}  Test Accuracy: {test_result['val_acc']:.4f}")