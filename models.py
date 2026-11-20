"""
q1_distillation/models.py
-------------------------
The teacher and the three student networks, all written from scratch for 32x32 CIFAR images.

Every model has the same interface:
    model(x)                 -> logits                        (normal use, ONNX export)
    model(x, with_hint=True) -> (logits, hint_feature_map)    (feature-based distillation)

The "hint" is an intermediate feature map at 4x4 resolution. The teacher's hint is the output
of its 3rd stage (1024 channels, 4x4); each student gives its own 4x4 feature map, and the
distillation framework learns a 1x1 conv "adapter" that maps it to the teacher's 1024 channels.

ImageNet versions of these networks start with a stride-2 stem (and ResNet-50 also with a
max-pool) because their images are 224x224. CIFAR images are only 32x32, so every model here
uses a stride-1 3x3 stem. All of them reach a 4x4 feature map (where the hint is taken).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================ teacher: ResNet-50 ===
class Bottleneck(nn.Module):
    """1x1 reduce -> 3x3 -> 1x1 expand (x4), with a residual connection."""
    expansion = 4

    def __init__(self, cin, width, stride):
        super().__init__()
        cout = width * self.expansion
        self.conv1 = nn.Conv2d(cin, width, 1, bias=False); self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False); self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, cout, 1, bias=False); self.bn3 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))


class ResNet50(nn.Module):
    """ResNet-50: [3, 4, 6, 3] bottleneck blocks, 23.7 M parameters.
    CIFAR stem: 3x3 conv (stride 1) + 2x2 max-pool, so the stages run at 16, 8, 4 and 2 px."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True), nn.MaxPool2d(2))
        cin, layers = 64, []
        for width, n, stride in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
            blocks = []
            for i in range(n):
                blocks.append(Bottleneck(cin, width, stride if i == 0 else 1))
                cin = width * 4
            layers.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.fc = nn.Linear(2048, num_classes)
        self.hint_channels = 1024

    def forward(self, x, with_hint=False):
        x = self.layer2(self.layer1(self.stem(x)))
        hint = self.layer3(x)                          # 1024 x 4 x 4  (teacher hint)
        x = self.layer4(hint)
        logits = self.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))
        return (logits, hint) if with_hint else logits


# ============================================================ student 1: MobileNetV2 ===
class InvertedResidual(nn.Module):
    """MobileNetV2 block: 1x1 expand -> 3x3 depthwise -> 1x1 linear projection (+ skip)."""

    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        hidden = cin * expand
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden),
                   nn.ReLU6(inplace=True),
                   nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)
        self.use_skip = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.block(x) if self.use_skip else self.block(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 (width 1.0). Settings (expand t, channels c, repeats n, stride s) follow the
    paper; for CIFAR only the stem uses stride 1 instead of 2."""
    CFG = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes=100):
        super().__init__()
        layers = [nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True))]
        cin = 32
        for t, c, n, s in self.CFG:
            for i in range(n):
                layers.append(InvertedResidual(cin, c, s if i == 0 else 1, t))
                cin = c
        n_hint = 1 + sum(n for _, _, n, _ in self.CFG[:5])   # stem + blocks up to the 96-channel group
        self.features_a = nn.Sequential(*layers[:n_hint])     # ... 96 channels, 4x4
        self.features_b = nn.Sequential(*layers[n_hint:])     # 160 and 320 channels, 2x2
        self.head = nn.Sequential(nn.Conv2d(320, 1280, 1, bias=False), nn.BatchNorm2d(1280), nn.ReLU6(inplace=True))
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
        self.hint_channels = 96

    def forward(self, x, with_hint=False):
        hint = self.features_a(x)                        # 96 x 4 x 4
        x = torch.flatten(F.adaptive_avg_pool2d(self.head(self.features_b(hint)), 1), 1)
        logits = self.classifier(x)
        return (logits, hint) if with_hint else logits


# ============================================================ student 2: SqueezeNet ===
class Fire(nn.Module):
    """SqueezeNet Fire module: 1x1 'squeeze' then parallel 1x1 and 3x3 'expand', concatenated."""

    def __init__(self, cin, squeeze, expand):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(cin, squeeze, 1), nn.BatchNorm2d(squeeze), nn.ReLU(inplace=True))
        self.e1 = nn.Sequential(nn.Conv2d(squeeze, expand, 1), nn.BatchNorm2d(expand), nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(squeeze, expand, 3, padding=1), nn.BatchNorm2d(expand), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.e1(x), self.e3(x)], 1)


class SqueezeNet(nn.Module):
    """SqueezeNet 1.1 layout (8 Fire modules, no fully-connected layer: the classifier is a 1x1
    conv + global average pooling). BatchNorm is added after each conv so it trains reliably."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                     # 16x16
            Fire(64, 16, 64), Fire(128, 16, 64),
            nn.MaxPool2d(2),                                     # 8x8
            Fire(128, 32, 128), Fire(256, 32, 128),
            nn.MaxPool2d(2),                                     # 4x4
            Fire(256, 48, 192), Fire(384, 48, 192), Fire(384, 64, 256), Fire(512, 64, 256))
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Conv2d(512, num_classes, 1))
        self.hint_channels = 512

    def forward(self, x, with_hint=False):
        hint = self.features(x)                          # 512 x 4 x 4
        logits = torch.flatten(F.adaptive_avg_pool2d(self.classifier(hint), 1), 1)
        return (logits, hint) if with_hint else logits


# ============================================================ student 3: EfficientNet-Lite0 ===
class MBConv(nn.Module):
    """EfficientNet-Lite MBConv: 1x1 expand -> kxk depthwise -> 1x1 projection (+ skip).
    'Lite' = no squeeze-and-excitation and ReLU6 instead of Swish (friendlier to edge hardware)."""

    def __init__(self, cin, cout, k, stride, expand):
        super().__init__()
        hidden = cin * expand
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(hidden, hidden, k, stride, k // 2, groups=hidden, bias=False), nn.BatchNorm2d(hidden),
                   nn.ReLU6(inplace=True), nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)
        self.use_skip = stride == 1 and cin == cout

    def forward(self, x):
        return x + self.block(x) if self.use_skip else self.block(x)


class EfficientNetLite0(nn.Module):
    """EfficientNet-Lite0: the B0 layout (width 1.0, depth 1.0) with SE removed and ReLU6;
    stem (32 ch) and head (1280 ch) are fixed as in the Lite family. CIFAR: stem stride 1."""
    # (kernel, stride, expand, out_channels, repeats)
    CFG = [(3, 1, 1, 16, 1), (3, 2, 6, 24, 2), (5, 2, 6, 40, 2), (3, 2, 6, 80, 3),
           (5, 1, 6, 112, 3), (5, 2, 6, 192, 4), (3, 1, 6, 320, 1)]

    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True))
        cin, stages = 32, []
        for k, s, e, c, n in self.CFG:
            blocks = []
            for i in range(n):
                blocks.append(MBConv(cin, c, k, s if i == 0 else 1, e))
                cin = c
            stages.append(nn.Sequential(*blocks))
        self.stages_a = nn.Sequential(*stages[:5])      # ... up to the 112-channel stage (4x4)
        self.stages_b = nn.Sequential(*stages[5:])      # 192 and 320 channels (2x2)
        self.head = nn.Sequential(nn.Conv2d(320, 1280, 1, bias=False), nn.BatchNorm2d(1280), nn.ReLU6(inplace=True))
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))
        self.hint_channels = 112

    def forward(self, x, with_hint=False):
        hint = self.stages_a(self.stem(x))               # 112 x 4 x 4
        x = self.head(self.stages_b(hint))
        logits = self.classifier(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))
        return (logits, hint) if with_hint else logits


STUDENTS = {"mobilenetv2": MobileNetV2, "squeezenet": SqueezeNet, "efficientnet_lite0": EfficientNetLite0}
STUDENT_NAMES = {"mobilenetv2": "MobileNetV2", "squeezenet": "SqueezeNet", "efficientnet_lite0": "EfficientNet-Lite0"}


def build(name, num_classes=100):
    return ResNet50(num_classes) if name == "resnet50" else STUDENTS[name](num_classes)
