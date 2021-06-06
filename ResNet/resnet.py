import torch
import torch.nn as nn

class ConvBlock(nn.Module):
	def __init__(self, in_filters, out_filters, kernel_size, stride, padding):
		super().__init__()
		self.expansion = 4
		self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride, padding=padding)
		self.bn = nn.BatchNorm2d(out_filters)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)

		return x

class IdentityBlock(nn.Module):

	def __init__(self, in_filters, out_filters, deep=True):
		super().__init__()

		self.deep = deep

		if self.deep is True:

			self.conv1 = ConvBlock(in_filters, out_filters, kernel_size=1, stride=1, padding=0)
			self.conv2 = ConvBlock(out_filters, out_filters, kernel_size=3, stride=1, padding=1)
			self.conv3 = nn.Conv2d(out_filters, out_filters*4, kernel_size=1, stride=1, padding=0)
			self.bn = nn.BatchNorm2d(out_filters*4)
			

		else:
			self.conv4 = ConvBlock(in_filters, out_filters, kernel_size=3, stride=1, padding=1)
			self.conv5 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1)
			self.bn_shallow = nn.BatchNorm2d(out_filters)

		self.relu = nn.ReLU()

	def forward(self, x):

		identity = x

		if self.deep is False:
			x = self.conv4(x)
			x = self.conv5(x)
			x = self.bn_shallow(x)
			x += identity
			x = self.relu(x)

			return x

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.bn(x)		
		x += identity
		x = self.relu(x)
		
		


		return x

class IdentityConvBlock(nn.Module):

	def __init__(self, in_filters, out_filters, stride, deep=True):
		super().__init__()
		self.deep = deep

		if self.deep is True:
			self.expansion = 4
			self.conv1 = ConvBlock(in_filters, out_filters, kernel_size=1, stride=1, padding=0)
			self.conv2 = ConvBlock(out_filters, out_filters, kernel_size=3, stride=stride, padding=1)

			self.conv3 = nn.Conv2d(out_filters, out_filters*self.expansion, kernel_size=1, stride=1, padding=0)
			self.bn = nn.BatchNorm2d(out_filters*self.expansion)

			self.identityConv =  nn.Sequential(
				nn.Conv2d(in_filters, out_filters*4, kernel_size=1, stride=stride), 
				nn.BatchNorm2d(out_filters*4)
				)

		else:
			self.conv4 = ConvBlock(in_filters, out_filters, kernel_size=3, stride=stride, padding=1)
			self.conv5 = nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1)
			self.bn_shallow = nn.BatchNorm2d(out_filters)
			self.identityConv_shallow = nn.Sequential(
			nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=stride), 
			nn.BatchNorm2d(out_filters)
			)

		self.relu = nn.ReLU()
	
	def forward(self, x):

		identity = x

		if self.deep is False:
			x = self.conv4(x)
			x = self.conv5(x)
			x = self.bn_shallow(x)
			identity = self.identityConv_shallow(identity)
			x += identity
			x = self.relu(x)

			return x

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.bn(x)
		identity = self.identityConv(identity)
		x += identity
		x = self.relu(x)
		
		return x

class ResNet(nn.Module):

	def __init__(self, num_blocks, image_channels=3, num_classes=10, deep=True):

		super().__init__()

		self.conv1 = ConvBlock(image_channels, 64, kernel_size=7, stride=2, padding=3)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		if deep is False:
			self.block1 = self._construct_residual_block_shallow(64, 64, num_blocks[0], stride=1)
			self.block2 = self._construct_residual_block_shallow(64, 128, num_blocks[1], stride=2)
			self.block3 = self._construct_residual_block_shallow(128, 256, num_blocks[2], stride=2)
			self.block4 = self._construct_residual_block_shallow(256, 512, num_blocks[3], stride=2)
			self.fc = nn.Linear(512, num_classes)

		else:
			self.block1 = self._construct_residual_block(64, 64, num_blocks[0], stride=1)
			self.block2 = self._construct_residual_block(256, 128, num_blocks[1], stride=2)
			self.block3 = self._construct_residual_block(512, 256, num_blocks[2], stride=2)
			self.block4 = self._construct_residual_block(1024, 512, num_blocks[3], stride=2)
			self.fc = nn.Linear(512 * 4, num_classes)

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
			

	def _construct_residual_block(self, input_filter_size, output_filter_size, num_layers, stride):

		layers = [IdentityConvBlock(in_filters=input_filter_size, out_filters=output_filter_size, stride=stride)]

		for _ in range(num_layers-1):
			layers.append(IdentityBlock(in_filters=output_filter_size*4, out_filters=output_filter_size))

		return nn.Sequential(*layers)

	def _construct_residual_block_shallow(self, input_filter_size, output_filter_size, num_layers, stride):
		layers = [IdentityConvBlock(in_filters=input_filter_size, out_filters=output_filter_size, stride=stride, deep=False)]

		for _ in range(num_layers-1):
			layers.append(IdentityBlock(in_filters=output_filter_size, out_filters=output_filter_size, deep=False))

		return nn.Sequential(*layers)

	def forward(self, x):

		x = self.conv1(x)
		x = self.maxpool(x)

		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = self.block4(x)

		x = self.avgpool(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc(x)

		return x


def resnet18(num_blocks=[2, 2, 2, 2]):

	return ResNet(num_blocks, deep=False)

def resnet34(num_blocks=[3, 4, 6, 3]):

	return ResNet(num_blocks, deep=False)

def resnet50(num_blocks=[3, 4, 6, 3]):

	return ResNet(num_blocks)

def resnet101(num_blocks=[3, 4, 23, 3]):

	return ResNet(num_blocks)

def resnet152(num_blocks=[3, 8, 36, 3]):

	return ResNet(num_blocks)



def test():

	device = 'cuda' # or 'cpu'

	model_18 = resnet18().to(device)
	model_34 = resnet34().to(device)
	model_50 = resnet50().to(device)
	model_101 = resnet101().to(device)
	model_152 = resnet152().to(device)

	x = torch.randn(2, 3, 224, 224).to(device)

	out_18 = model_18(x)
	out_34 = model_34(x)
	out_50 = model_50(x)
	out_101 = model_101(x)
	out_152 = model_152(x)

	print(out_18.shape)
	print(out_34.shape)
	print(out_50.shape)
	print(out_101.shape)
	print(out_152.shape)


if __name__ == '__main__':
	test()