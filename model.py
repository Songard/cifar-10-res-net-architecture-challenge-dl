"""CIFAR-size residual networks, trained without pretrained weights.

Architecture reference: He et al., https://arxiv.org/abs/1512.03385
CIFAR adaptation reference: https://github.com/kuangliu/pytorch-cifar
The baseline is a half-width ResNet-18 (width 32). Other entries in ARCHITECTURES
change stage widths and depths. The default residual block is
ReLU(S(x) + F(x)) with F = conv-BN-ReLU-conv-BN; preact-highres instead uses
the full pre-activation variant from https://arxiv.org/abs/1603.05027.
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


class PreActBlock(nn.Module):
    """Full pre-activation basic block, He et al. (2016), arXiv:1603.05027.

    Identity shortcuts carry x; dimension-changing projections use the pre-activated input.
    Implemented here from the paper, without copying a third-party implementation.
    """

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.shortcut = None if stride == 1 and in_channels == channels else nn.Conv2d(
            in_channels, channels, 1, stride, bias=False)

    def forward(self, x):
        h = torch.relu(self.bn1(x))
        shortcut = x if self.shortcut is None else self.shortcut(h)
        h = self.conv1(h)
        return self.conv2(torch.relu(self.bn2(h))) + shortcut


class CifarResNet(nn.Module):
    """3x3 stem, residual stages (the first block of stages 2+ halves resolution), average pool, linear head.

    `CifarResNet(width)` is the original baseline with widths (w, 2w, 4w, 8w) and two blocks per stage;
    module names are unchanged so older checkpoints still load.
    """

    def __init__(self, width=32, widths=None, depths=(2, 2, 2, 2), preactivation=False):
        super().__init__()
        widths = list(widths or [width, 2 * width, 4 * width, 8 * width])
        if len(widths) != len(depths):
            raise ValueError("widths and depths must have the same length")
        self.width = widths[0]
        self.preactivation = preactivation
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, padding=1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU())
        if preactivation:
            self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, padding=1, bias=False))
        block_type = PreActBlock if preactivation else BasicBlock
        blocks = []
        previous = widths[0]
        for stage, (channels, depth) in enumerate(zip(widths, depths)):
            for block in range(depth):
                stride = 2 if stage > 0 and block == 0 else 1
                blocks.append(block_type(previous, channels, stride))
                previous = channels
        self.blocks = nn.Sequential(*blocks)
        self.final_activation = nn.Sequential(nn.BatchNorm2d(previous), nn.ReLU()) if preactivation else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(previous, 10)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        h = self.final_activation(self.blocks(self.stem(x)))
        return self.head(self.pool(h).flatten(1))


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
    # The same layout at width 68 (4,884,926 parameters, 98% of the budget); trained once, on all 50,000 images.
    "resnet20-w68": dict(widths=(68, 136, 272), depths=(3, 3, 3)),
    # Move capacity toward the 32x32 and 16x16 stages; retain the original block type.
    "resnet-highres": dict(widths=(96, 160, 224), depths=(3, 3, 2)),
    # Matched widths/depths isolate full pre-activation from the capacity redistribution.
    "preact-highres": dict(widths=(96, 160, 224), depths=(3, 3, 2), preactivation=True),
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
