import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from LeNet import LeNet
from tqdm import tqdm
from torch.autograd import Variable

def check_accuracy(test_loader, model, device, train=False):

	correct = 0
	total = 0
	model.eval()
	
	with torch.no_grad():

		for images, labels in tqdm(test_loader):
		    
		    images = images.to(device=device)
		    labels = labels.to(device=device)
		    y = model(images)
		    
		    predictions = torch.argmax(y, dim=1)

		    correct += torch.sum((predictions == labels).float())
		    total += y.size(0)

		if train:
			mode = 'Train Accuracy'
		else:
			mode = 'Test Accuracy'

		print('{}: {}'.format(mode, correct/total))
	



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

LEARNING_RATE = 0.01
NUM_EPOCHS = 10

train_dataset = datasets.MNIST(root='dataset/', train=True, 
                               transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]), download=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, 
                              transform=transforms.Compose([transforms.Pad(2), transforms.ToTensor()]), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


model = LeNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in range(NUM_EPOCHS):

	print(f"Training --- Epoch:- {epoch+1}/{NUM_EPOCHS}")

	model.train()
	for data, targets in tqdm(train_loader):

		data = data.to(device)
		targets = targets.to(device)

		optimizer.zero_grad()

		predictions = model(data)
		loss = criterion(predictions, targets)

		loss.backward()
		optimizer.step()



check_accuracy(train_loader, model, device, train=True)
check_accuracy(test_loader, model, device)