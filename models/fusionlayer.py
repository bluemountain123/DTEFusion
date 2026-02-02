import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.conv3 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)      
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # First convolutional layer
        out1 = self.relu(self.bn1(self.conv1(x)))
        # print("out1",out1.shape)
        # Second convolutional layer
        out2 = self.relu(self.bn2(self.conv2(out1)))
        # print("out2",out2.shape)
        # Concatenate outputs of the first two layers
        temp = torch.cat([out1, out2], dim=1)
        # print("temp",temp.shape)
        # Third convolutional layer, using concatenated result
        out3 = self.relu(self.bn3(self.conv3(temp)))
        # print("out3",out3.shape)
        return out3

class ModularNetwork(nn.Module):
    def __init__(self, in_channels, num_blocks, kernel_size, stride):
        super(ModularNetwork, self).__init__()
        blocks = []
        out_channels = 48  # Initial output channels
        # Stack multiple DenseBlocks
        for _ in range(num_blocks):
            blocks.append(DenseBlock(in_channels, out_channels, kernel_size, stride))
            # Update input channels to current block's output channels
            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        # print("x_shape_beforeFL", x.shape)
        x = self.blocks(x)
        # print("x_afterBlocks",x.shape)
        return x

# Example usage
if __name__ == "__main__":
    # Test network structure
    model = ModularNetwork(in_channels=1, num_blocks=3, kernel_size=3, stride=1)
    sample_input = torch.randn(1, 1, 64, 64)  # 假设输入是 (1, 1, 64, 64)
    output = model(sample_input)
    print(output.shape)