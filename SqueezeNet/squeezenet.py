import torch
import torch.nn as nn


class FireModule(nn.Module):

    def __init__(self, in_channels, s1, fe1 , fe3):

        super().__init__()

        self.sq = nn.Conv2d(in_channels, s1, kernel_size=1)
        self.e1 = nn.Conv2d(s1, fe1, kernel_size=1)
        self.e3 = nn.Conv2d(s1, fe3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.sq(x))
        x1 = self.relu(self.e1(x))
        x2 = self.relu(self.e3(x))

        x_concat = torch.cat([x1, x2], dim=1)

        return x_concat

class SqueezeNet(nn.Module):

    def __init__(self, img_channels=3, num_classes=10, version="v1"):

        super().__init__()

        self._version = version

        self.conv1 = nn.Conv2d(in_channels=img_channels, out_channels=96, kernel_size=7, stride=2, padding=2)
        
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire2 = FireModule(96, 16, 64, 64)
        self.fire3 = FireModule(128, 16, 64, 64)
        self.fire4 = FireModule(128, 32, 128, 128)

        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire5 = FireModule(256, 32, 128, 128)
        self.fire6 = FireModule(256, 48, 192, 192)
        self.fire7 = FireModule(384, 48, 192, 192)
        self.fire8 = FireModule(384, 64, 256, 256)

        self.max_pool8 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire9 = FireModule(512, 64, 256, 256)

        self.conv10 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1)

        self.avg_pool = nn.AvgPool2d(kernel_size=13, stride=1)

        self.flatten = nn.Flatten()

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        if self._version == "v1":

            x = self.conv1(x)
            x = self.max_pool1(x)
            x = self.fire2(x)
            x = self.fire3(x)
            x = self.fire4(x)
            x = self.max_pool4(x)
            x = self.fire5(x)
            x = self.fire6(x)
            x = self.fire7(x)
            x = self.fire8(x)
            x = self.max_pool8(x)
            x = self.fire9(x)

            x = self.dropout(x)

            x = self.conv10(x)

            x = self.avg_pool(x)

            x = self.flatten(x)

            x = self.softmax(x)

            return x
        
        else:
            raise(f"Version should be either \"v1\" or \"v2\" but entered version is {self._version}")


if __name__ == "__main__":

    test_input = torch.rand((4, 3, 224, 224))
    test_num_classes = 10

    model = SqueezeNet()

    test_output = model(test_input)
    print(test_output.shape)

        

