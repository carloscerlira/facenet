import torch
import torch.nn as nn
from torchvision.models import resnet50

from torch.utils.model_zoo import load_url as load_state_dict_from_url

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class FaceNetModel(nn.Module):
    def __init__(self, pretrained=False):
        super(FaceNetModel, self).__init__()

        self.model = resnet50(pretrained)
        embedding_size = 128
        num_classes = 500
        self.cnn = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4)

        self.model.fc = nn.Sequential(
            Flatten(),
            nn.Linear(100352, embedding_size))

        self.model.classifier = nn.Linear(embedding_size, num_classes)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.cnn(x)
        x = self.model.fc(x)

        features = self.l2_norm(x)
        alpha = 10
        features = features * alpha
        return features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res

model_url = 'https://github.com/carloscerlira/facenet/releases/download/facenet/iimas_model.pth'

# def load_state(url):
#     state = load_state_dict_from_url(url, progress=True)
#     return state

# def get_iimas_model(pretrained=True):
#     model = FaceNetModel()
#     if pretrained:
#         state = load_state(model_url)
#         model.load_state_dict(state['state_dict'])
#     return model

def get_iimas_model(pretrained=True):
    model = FaceNetModel()
    if pretrained:
        state = torch.load('./logs/iimas_model.pth')
        model.load_state_dict(state["state_dict"])
    return model
