import torch
import torch.nn as nn

class ConvBlock(nn.Module):

	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):

		super().__init__()
		self.relu = nn.ReLU()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding=padding)
		self.batchnorm = nn.BatchNorm2d(out_channels)


	def forward(self, x, batchnorm=False):
		if batchnorm is True:
			return self.relu(self.batchnorm(self.conv(x)))
		return self.relu(self.conv(x))


class InceptionBlock(nn.Module):

	def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
		super().__init__()

		self.branch_1 = ConvBlock(in_channels, out_1x1, kernel_size=1, stride=1, padding=0)

		self.branch_2 = nn.Sequential(
			ConvBlock(in_channels, red_3x3, kernel_size=1, stride=1, padding=0),
			ConvBlock(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1)
			)

		self.branch_3 = nn.Sequential(
			ConvBlock(in_channels, red_5x5, kernel_size=1, stride=1, padding=0),
			ConvBlock(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2)
			)

		self.branch_4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			ConvBlock(in_channels, out_pool, kernel_size=1, stride=1, padding=0)
			)


	def forward(self, x):

		return torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], axis=1)


class GoogLeNet(nn.Module):

	def __init__(self, in_channels = 3, num_classes = 10):
		super().__init__()

		self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
		self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.conv2 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
		self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		# For ref:- in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool

		self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
		self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

		self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
		self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
		self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
		self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
		self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

		self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
		self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

		self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

		self.dropout = nn.Dropout(p=0.4)

		self.fc1 = nn.Linear(in_features=1024, out_features=num_classes)

	def forward(self, x):

		x = self.conv1(x)
		x = self.maxpool1(x)
		x = self.conv2(x)
		x = self.maxpool2(x)

		x = self.inception_3a(x)
		x = self.inception_3b(x)

		x = self.maxpool3(x)

		x = self.inception_4a(x)
		x = self.inception_4b(x)
		x = self.inception_4c(x)
		x = self.inception_4d(x)
		x = self.inception_4e(x)

		x = self.maxpool4(x)

		x = self.inception_5a(x)
		x = self.inception_5b(x)

		x = self.avgpool(x)

		x = x.reshape(x.shape[0], -1)

		x = self.dropout(x)

		x = self.fc1(x)

		return x


if __name__ == '__main__':

	sample_input = torch.randn(5, 3, 224, 224)
	model = GoogLeNet()

	# Should Output:- 5 x 10
	print(model(sample_input).shape)
