import torch
import config
import torchvision
import torch.nn as nn
import models


class BaseFeatureNet(nn.Module):
    def __init__(self, base_model_name=models.VGG13, pretrained=True):
        super(BaseFeatureNet, self).__init__()
        base_model_name = base_model_name.upper()
        self.fc_features = None

        if base_model_name == models.VGG13:
            base_model = torchvision.models.vgg13(pretrained=pretrained)
            self.feature_len = 4096
            self.features = base_model.features
            self.fc_features = nn.Sequential(*list(base_model.classifier.children())[:-1])

        elif base_model_name == models.VGG11BN:
            base_model = torchvision.models.vgg11_bn(pretrained=pretrained)
            self.feature_len = 4096
            self.features = base_model.features
            self.fc_features = nn.Sequential(*list(base_model.classifier.children())[:-1])

        elif base_model_name == models.VGG13BN:
            base_model = torchvision.models.vgg13_bn(pretrained=pretrained)
            self.feature_len = 4096
            self.features = base_model.features
            self.fc_features = nn.Sequential(*list(base_model.classifier.children())[:-1])

        elif base_model_name == models.ALEXNET:
            # base_model = torchvision.models.alexnet(pretrained=pretrained)
            base_model = torchvision.models.alexnet(pretrained=pretrained)
            self.feature_len = 4096
            self.features = base_model.features
            self.fc_features = nn.Sequential(*list(base_model.classifier.children())[:-1])

        elif base_model_name == models.RESNET50:
            base_model = torchvision.models.resnet50(pretrained=pretrained)
            self.feature_len = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-1])
        elif base_model_name == models.RESNET101:
            base_model = torchvision.models.resnet101(pretrained=pretrained)
            self.feature_len = 2048
            self.features = nn.Sequential(*list(base_model.children())[:-1])

        elif base_model_name == models.INCEPTION_V3:
            base_model = torchvision.models.inception_v3(pretrained=pretrained)
            base_model_list = list(base_model.children())[0:13]
            base_model_list.extend(list(base_model.children())[14:17])
            self.features = nn.Sequential(*base_model_list)
            self.feature_len = 2048

        else:
            raise NotImplementedError(f'{base_model_name} is not supported models')

    def forward(self, x):
        # x = x[:,0]
        # if len(x.size()) == 5:
        batch_sz = x.size(0)
        view_num = x.size(1)
        x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))

        with torch.no_grad():
            x = self.features[:1](x)
        x = self.features[1:](x)

        x = x.view(x.size(0), -1)
        x = self.fc_features(x) if self.fc_features is not None else x

        # max view pooling
        x_view = x.view(batch_sz, view_num, -1)
        x, _ = torch.max(x_view, 1)

        return x, x_view


class BaseClassifierNet(nn.Module):
    def __init__(self, base_model_name=models.VGG13, num_classes=40, pretrained=True):
        super(BaseClassifierNet, self).__init__()
        base_model_name = base_model_name.upper()
        if base_model_name in (models.VGG13, models.VGG13BN, models.ALEXNET, models.VGG11BN):
            self.feature_len = 4096
        elif base_model_name in (models.RESNET50, models.RESNET101, models.INCEPTION_V3):
            self.feature_len = 2048
        else:
            raise NotImplementedError(f'{base_model_name} is not supported models')

        self.classifier = nn.Linear(self.feature_len, num_classes)

    def forward(self, x):
        x = self.classifier(x)
        return x


class MVCNN(nn.Module):
    def __init__(self, pretrained=True):
        super(MVCNN, self).__init__()
        base_model_name = config.base_model_name
        num_classes = config.view_net.num_classes
        print(f'\ninit {base_model_name} model...\n')
        self.features = BaseFeatureNet(base_model_name, pretrained)
        self.classifier = BaseClassifierNet(base_model_name, num_classes, pretrained)

    def forward(self, x):
        x, _ = self.features(x)
        x = self.classifier(x)
        return x


