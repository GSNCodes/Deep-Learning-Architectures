import torch
import torch.nn as nn


class VGG(nn.Module):

	def __init__(self, vgg_type, in_channels=3, num_classes=10):

		super().__init__()
		self.in_channels = in_channels

		self.VGG16_architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
		self.VGG19_architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
		
		if vgg_type == 16:
			self.conv_layers = self.create_conv_layers(self.VGG16_architecture, in_channels)

		elif vgg_type == 19:
			self.conv_layers = self.create_conv_layers(self.VGG19_architecture, in_channels)

		self.fcs = nn.Sequential(
			nn.Linear(512*7*7, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(),
			nn.Dropout(p=0.5),
			nn.Linear(4096, num_classes))

	def forward(self, x):
		x = self.conv_layers(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fcs(x)

		return x

	def create_conv_layers(self, architecture, in_channels):
		layers = []
		batch_norm = True
		for x in architecture:
			if type(x) == int:

				out_channels = x
				layers += [
							nn.Conv2d(in_channels=in_channels, 
							out_channels=out_channels, 
							kernel_size=3, 
							stride=1, 
							padding=1)
							]
				
				if batch_norm:
					layers.append(nn.BatchNorm2d(x))
				
				layers.append(nn.ReLU())

				in_channels = x


			elif x == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

		return nn.Sequential(*layers)


def vgg16(in_channels=3, num_classes=10):

	return VGG(16, in_channels, num_classes)

def vgg19(in_channels=3, num_classes=10):

	return VGG(19, in_channels, num_classes)

if __name__ == '__main__':

	model_vgg_16 = vgg16()
	x = torch.randn(1, 3, 224, 224)
	print("Test - VGG16")
	print("Output Shape:", (model_vgg_16(x).shape))
	print("Expected Shape: (with default num_classes of 10)", [1, 10])


	model_vgg_19 = vgg19()
	x = torch.randn(1, 3, 224, 224)
	print("Test - VGG19")
	print("Output Shape:", model_vgg_19(x).shape)
	print("Expected Shape: (with default num_classes of 10)", [1, 10])


