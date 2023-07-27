import torch
import torch.nn as nn
import torch.functional as F

class ConvBlock(nn.Module):
    expansion = 1

    def __init__(self, input_channels, output_channels,stride = 1):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size= 3, stride=stride, bias= False
        )
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, stride= 1, padding= 1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or input_channels != self.expansion * output_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_channels, self.expansion * output_channels, kernel_size=1, stride= stride, bias= False),
                nn.BatchNorm2d(self.expansion * output_channels)
            )



