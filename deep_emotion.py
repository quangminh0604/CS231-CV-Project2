import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Deep_Emotion(nn.Module):
    def __init__(self):
        super(Deep_Emotion, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)  # Thêm dropout nhẹ cho conv layers

        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 128)  # Thêm khối residual thứ hai
        self.conv_dilated = nn.Conv2d(128, 128, 3, padding=2, dilation=2)  # Dilated convolution
        self.bn_dilated = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 7)
        self.dropout = nn.Dropout(0.6)  # Tăng dropout

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(16*12*12, 32),  # Điều chỉnh dựa trên kích thước đầu ra
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        #print("Shape after localization:", xs.shape)  # Debug: In kích thước tensor
        xs = xs.view(-1, 16*12*12)  # Điều chỉnh dựa trên kích thước đầu ra
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x

    def forward(self, input):
        out = self.stn(input)

        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool2(out)
        out = self.dropout1(out)

        out = self.res_block1(out)
        out = self.res_block2(out)
        out = F.relu(self.bn_dilated(self.conv_dilated(out)))
        out = self.pool4(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(-1, 128)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out