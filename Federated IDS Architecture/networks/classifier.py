import torch
import torch.nn as nn 
from networks.resnet import resnet18

class Classifier(nn.Module):
    def __init__(self, num_classes) -> None:
        super(Classifier, self).__init__()
        self.encoder = resnet18() 
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        return self.fc(self.encoder(x))