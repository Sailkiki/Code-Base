import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    
    def __init__(self, in_planes, ratio=16):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False),
        )


    def forward(self, x):
        # x: (B, C, N)

        avg_out = torch.mean(x, dim=2, keepdim=True)
        max_out = torch.max(x, dim=2, keepdim=True)[0]

        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)

        attention = torch.sigmoid(avg_out + max_out)

        return x * attention.expand_as(x)



class SpatialAttention(nn.Module):
    
    def __init__(self, kernel_size=7) -> None:
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2
        
        # 这里的卷积是沿着点的维度(N)进行的，用来捕捉“空间”或点间的关系
        # 输入是2个通道（平均池化和最大池化），输出是1个通道（注意力权重）
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)

        
    def forward(self, x):

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]

        y = torch.cat([avg_out, max_out], dim=1)

        attention = torch.sigmoid(self.conv1(y))

        return x * attention.expand_as(x)
    


class PointCBAM(nn.Module):
    def __init__(self, in_planes, ratios=16, kernel_size=7) -> None:
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratios)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):

        return self.sa(self.ca(x))
    

class splepointnet(nn.Module):
    def __init__(self, num_class=10) -> None:
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
        )

        self.cbam = PointCBAM(in_planes=128)

        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )

        self.classfier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        # x: (B, 3, N)
        x = self.mlp1(x)
        x = self.cbam(x)
        x = self.mlp2(x)
        # x.shape = (B, 256, N)

        # Global max pooling
        x = torch.max(x, dim=2, keepdim=False)[0]
        # x.shape = (B, 256)

        # Classifier
        x = self.classfier(x)
        return x


if __name__ == '__main__':

    model = splepointnet()
    
    x = torch.randn(1, 3, 1024)

    y = model(x)

    print(y.shape)