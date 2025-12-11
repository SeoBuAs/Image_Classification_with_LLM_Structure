import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

#### Normal Extractor #############################################################################
class CustomFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, output_dim=400):
        super(CustomFeatureExtractor, self).__init__()
        self.conv1x1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_channels * 448 * 448, output_dim)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children()))
        self.fc = nn.Linear(25088 * 2 * 2, 400)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class DenseNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(DenseNetFeatureExtractor, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.fc = nn.Linear(1024 * 14 * 14, 400)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ConvNextBaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvNextBaseFeatureExtractor, self).__init__()
        convnext = models.convnext_base(pretrained=True)
        self.features = nn.Sequential(*list(convnext.children())[:-1])
        self.fc = nn.Linear(1024, 400)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNetBaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetBaseFeatureExtractor, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, 400)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class EfficientNetBaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetBaseFeatureExtractor, self).__init__()
        efficientnet = models.efficientnet_b4(pretrained=True)
        self.features = efficientnet.features
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1792, 400)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class MobileNetBaseFeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileNetBaseFeatureExtractor, self).__init__()
        mobilenet = models.mobilenet_v3_large(pretrained=True)
        self.features = mobilenet.features
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(960, 400)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

#### MultiScale Extractor #############################################################################
class ConvNeXtMultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvNeXtMultiScaleFeatureExtractor, self).__init__()
        convnext = models.convnext_base(pretrained=True)
        self.features = nn.Sequential(*list(convnext.children())[:-2])
        self.fc = nn.Linear(702464, 400)
        
    def forward(self, x):
        scales = [0.5, 1.0, 1.5]
        
        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class EfficientNetB4MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetB4MultiScaleFeatureExtractor, self).__init__()
        efficientnet_b4 = models.efficientnet_b4(pretrained=True)
        self.features = nn.Sequential(*list(efficientnet_b4.children())[:-1])
        self.fc = nn.Linear(5376, 400)
        
    def forward(self, x):
        scales = [0.5, 1.0, 1.5]
        
        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class MobileNetV3LargeMultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileNetV3LargeMultiScaleFeatureExtractor, self).__init__()
        mobilenet_v3 = models.mobilenet_v3_large(pretrained=True)
        self.features = nn.Sequential(*list(mobilenet_v3.children())[:-1])
        self.fc = nn.Linear(960 * 3, 400)  # Adjusted for multi-scale concatenation

    def forward(self, x):
        scales = [0.5, 1.0, 1.5]

        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class ResNet101MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet101MultiScaleFeatureExtractor, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(resnet101.children())[:-2])
        self.fc = nn.Linear(1404928, 400)  # Adjusted for multi-scale concatenation

    def forward(self, x):
        scales = [0.5, 1.0, 1.5]

        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class DenseNet121MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(DenseNet121MultiScaleFeatureExtractor, self).__init__()
        densenet121 = models.densenet121(pretrained=True)
        self.features = nn.Sequential(*list(densenet121.features.children()))
        self.fc = nn.Linear(702464, 400)  # Adjusted for multi-scale concatenation

    def forward(self, x):
        scales = [0.5, 1.0, 1.5]

        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class VGG16MultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16MultiScaleFeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features.children()))
        self.fc = nn.Linear(351232, 400)  # Adjusted for multi-scale concatenation

    def forward(self, x):
        scales = [0.5, 1.0, 1.5]

        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class CustomMultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, output_dim=400):
        super(CustomMultiScaleFeatureExtractor, self).__init__()
        self.conv1x1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2107392, output_dim)

    def forward(self, x):
        scales = [0.5, 1.0, 1.5]

        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            scaled_image = self.conv1x1(scaled_image)
            scaled_image = self.batch_norm(scaled_image)
            scaled_image = self.relu(scaled_image)
            scaled_image = self.flatten(scaled_image)
            features.append(scaled_image)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output


#### MoreMultiScale Extractor #############################################################################
class ConvNeXtMoreMultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvNeXtMoreMultiScaleFeatureExtractor, self).__init__()
        convnext = models.convnext_base(pretrained=True)
        self.features = nn.Sequential(*list(convnext.children())[:-2])
        self.fc = nn.Linear(1110016, 400)
        
    def forward(self, x):
        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        
        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class EfficientNetB4MoreMultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetB4MoreMultiScaleFeatureExtractor, self).__init__()
        efficientnet_b4 = models.efficientnet_b4(pretrained=True)
        self.features = nn.Sequential(*list(efficientnet_b4.children())[:-1])
        self.fc = nn.Linear(10752, 400)
        
    def forward(self, x):
        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
        
        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class MobileNetV3LargeMoreMultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileNetV3LargeMoreMultiScaleFeatureExtractor, self).__init__()
        mobilenet_v3 = models.mobilenet_v3_large(pretrained=True)
        self.features = nn.Sequential(*list(mobilenet_v3.children())[:-1])
        self.fc = nn.Linear(5760, 400)  # Adjusted for multi-scale concatenation

    def forward(self, x):
        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class ResNet101MoreMultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet101MoreMultiScaleFeatureExtractor, self).__init__()
        resnet101 = models.resnet101(pretrained=True)
        self.features = nn.Sequential(*list(resnet101.children())[:-2])
        self.fc = nn.Linear(2349056, 400)  # Adjusted for multi-scale concatenation

    def forward(self, x):
        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class DenseNet121MoreMultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(DenseNet121MoreMultiScaleFeatureExtractor, self).__init__()
        densenet121 = models.densenet121(pretrained=True)
        self.features = nn.Sequential(*list(densenet121.features.children()))
        self.fc = nn.Linear(1110016, 400)  # Adjusted for multi-scale concatenation

    def forward(self, x):
        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class VGG16MoreMultiScaleFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16MoreMultiScaleFeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features.children()))
        self.fc = nn.Linear(555008, 400)  # Adjusted for multi-scale concatenation

    def forward(self, x):
        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            feature_map = self.features(scaled_image)
            feature_map = torch.flatten(feature_map, 1)
            features.append(feature_map)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output

class CustomMoreMultiScaleFeatureExtractor(nn.Module):
    def __init__(self, input_channels=3, output_dim=400):
        super(CustomMoreMultiScaleFeatureExtractor, self).__init__()
        self.conv1x1 = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(input_channels)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3424512, output_dim)

    def forward(self, x):
        scales = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

        features = []
        for scale in scales:
            scaled_image = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
            scaled_image = self.conv1x1(scaled_image)
            scaled_image = self.batch_norm(scaled_image)
            scaled_image = self.relu(scaled_image)
            scaled_image = self.flatten(scaled_image)
            features.append(scaled_image)

        combined_features = torch.cat(features, dim=1)
        output = self.fc(combined_features)
        return output