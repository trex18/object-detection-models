import torch
import torch.nn as nn

YOLO_CONFIG = [
    # kernel_size, out_channels
    (7, 64, 2),
    'M',
    (3, 192, 1),
    'M',
    (1, 128, 1),
    (3, 256, 1),
    (1, 256, 1),
    (3, 512, 1),
    'M',
    (1, 256, 1),
    (3, 512, 1),
    (1, 256, 1),
    (3, 512, 1),
    (1, 256, 1),
    (3, 512, 1),
    (1, 256, 1),
    (3, 512, 1),
    (1, 512, 1),
    (3, 1024, 1),
    'M',
    (1, 512, 1),
    (3, 1024, 1),
    (1, 512, 1),
    (3, 1024, 1),
    (3, 1024, 1),
    (3, 1024, 2),
    (3, 1024, 1),
    (3, 1024, 1)
]

def conv_block(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )

class YOLOv1(nn.Module):
    def __init__(self, config, num_classes, hidden_fc_neurons=10):
        """
        :param config: list of either tuples of 3 (kernel_size, out_channels, stride) or 'M'
        :param inter_fc_neurons: the number of neurons in hidden FC layer
        """
        super(YOLOv1, self).__init__()
        self.config = config
        self.conv_layers = self._make_conv_layers()
        self.fc_layers = self._make_fc_layers(num_classes, hidden_fc_neurons)

    def _make_conv_layers(self):
        layers = []
        in_channels = 3
        for layer in self.config:
            assert (isinstance(layer, tuple) and len(layer) == 3) or layer == 'M'
            if layer == 'M':
                layers.append(nn.MaxPool2d(2, 2))
                continue
            kernel_size, out_channels, stride = layer
            # we set padding = kernel_size // 2 for same convolutions
            layers.append(conv_block(in_channels, out_channels, kernel_size, stride, kernel_size//2))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _make_fc_layers(self, num_classes, hidden_fc_neurons):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, hidden_fc_neurons),
            nn.BatchNorm1d(hidden_fc_neurons),
            nn.LeakyReLU(0.1),
            # as in the paper, we assume that every cell contains two boxes,
            # so there are 2 * 5 = 10 parameters for coordinates and confidence
            nn.Linear(hidden_fc_neurons, 7 * 7 * (num_classes + 10))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        out = self.fc_layers(x)
        return out

if __name__ == "__main__":
    DEVICE = torch.device("cuda")
    model = YOLOv1(YOLO_CONFIG, 5, 4).to(DEVICE)
    img = torch.rand(2, 3, 448, 448).to(DEVICE)
    result = model(img)
    print(result.shape)
