import torch
from torch import nn

# Input NxNx3 # For CIFAR 10, you can set img_size to 70
# Conv 11x11, 64 filters, stride 4, padding 2
# MaxPool 3x3, stride 2
# Conv 5x5, 192 filters, padding 2
# MaxPool 3x3, stride 2
# Conv 3x3, 384 filters, padding 1
# Conv 3x3, 256 filters, padding 1
# Conv 3x3, 256 filters, padding 1
# MaxPool 3x3, stride 2
# nn.AdaptiveAvgPool2d((6, 6)) # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
# flatten into a vector of length x # what is x?
# Dropout 0.5
# Linear with 4096 output units
# Dropout 0.5
# Linear with 4096 output units
# Linear with num_classes output units

class AlexNet(nn.Module):
    """Fake LeNet with 32x32 color images and 200 classes"""

    def __init__(self, num_classes: int = 200) -> None:
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6))
        )

        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        print(x.shape, x)
        x = self.classifier(x)
        return x
