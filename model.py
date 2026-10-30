"""CIFAR-size residual networks, trained without pretrained weights.

Architecture reference: He et al., https://arxiv.org/abs/1512.03385
CIFAR adaptation reference: https://github.com/kuangliu/pytorch-cifar
The baseline is a half-width ResNet-18 (width 32). Other entries in ARCHITECTURES
change only stage widths and the number of blocks per stage; every block is the
course's residual block ReLU(S(x) + F(x)) with F = conv-BN-ReLU-conv-BN.
"""

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        return torch.relu(self.bn2(self.conv2(h)) + self.shortcut(x))


class CifarResNet(nn.Module):
    """3x3 stem, residual stages (the first block of stages 2+ halves resolution), average pool, linear head.

    `CifarResNet(width)` is the original baseline with widths (w, 2w, 4w, 8w) and two blocks per stage;
    module names are unchanged so older checkpoints still load.
    """

    def __init__(self, width=32, widths=None, depths=(2, 2, 2, 2)):
        super().__init__()
        widths = list(widths or [width, 2 * width, 4 * width, 8 * width])
        if len(widths) != len(depths):
            raise ValueError("widths and depths must have the same length")
        self.width = widths[0]
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, padding=1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        blocks = []
        previous = widths[0]
        for stage, (channels, depth) in enumerate(zip(widths, depths)):
            for block in range(depth):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(BasicBlock(previous, channels, stride))
                previous = channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(previous, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return self.head(self.pool(self.blocks(self.stem(x))).flatten(1))


# Named, parameter-checked architectures. Checkpoints store the name, so inference rebuilds the same network.
ARCHITECTURES = {
    # Original baseline: half-width ResNet-18, 2,797,610 parameters.
    "resnet18-w32": dict(widths=(32, 64, 128, 256), depths=(2, 2, 2, 2)),
    # Uniformly widened ResNet-18, 4,815,940 parameters.
    "resnet18-w42": dict(widths=(42, 84, 168, 336), depths=(2, 2, 2, 2)),
    # Full-width ResNet-18 stages 1-3; stage 4 trimmed to one 352-channel block to fit the budget.
    "resnet18-64-352": dict(widths=(64, 128, 256, 352), depths=(2, 2, 2, 1)),
    # ResNet-20 layout (three stages of three blocks, He et al. Sec. 4.2) at four times the width, 4,327,754 parameters.
    "resnet20-x4": dict(widths=(64, 128, 256), depths=(3, 3, 3)),
}


def build_model(arch):
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {sorted(ARCHITECTURES)}")
    return CifarResNet(**ARCHITECTURES[arch])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def multiply_accumulates(model, size=32):
    """Multiply-accumulate operations of convolutions and linear layers for one size x size RGB image."""
    total = 0

    def count(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            total += output[0].numel() * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1]
        else:
            total += module.in_features * module.out_features

    hooks = [m.register_forward_hook(count) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size, device=next(model.parameters()).device))
    model.train(training)
    for hook in hooks:
        hook.remove()
    return total
