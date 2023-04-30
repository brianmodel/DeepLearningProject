from collections import OrderedDict

import torch
from torch import nn
from transformers import ASTModel

from transformers import AutoFeatureExtractor
from utils import SAMPLE_RATE
from models.ASTTransferClassifier import ASTPipeline
from models.CNNClassifier import CNNPipeline

class CombinedClassifier(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.ast_pipeline = ASTPipeline()
        self.cnn_pipeline = CNNPipeline()

        self.cnn = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1, 8, kernel_size=(4,4), stride=(1,1), padding=(2,2))),
          ('relu1', nn.ReLU()),
          ('bn1', nn.BatchNorm2d(8)),
          ('pool1', nn.MaxPool2d(4,4)),
          ('conv2', nn.Conv2d(8, 10, kernel_size=(3,3), stride=(1,1))),
          ('relu2', nn.ReLU()),
          ('bn2', nn.BatchNorm2d(10)),
          ('pool2', nn.MaxPool2d(3,3))
        ]))

        self.ast_model = ASTModel.from_pretrained('bookbot/distil-ast-audioset')
        for param in self.ast_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=0.7)
        self.linear1 = nn.Linear(768+480, 100)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)
    
    def forward(self, x):
        x_pretrained = self.ast_pipeline(x.detach())
        x_pretrained = x_pretrained.to(self.device)
        x_pretrained = self.ast_model(input_values=x_pretrained).pooler_output

        x_pretrained = self.dropout(x_pretrained)
        x = x.to(self.device)
        x_cnn = self.cnn_pipeline(x)
        if len(x_cnn.shape) == 3:
            x_cnn = torch.unsqueeze(x_cnn, dim=1)

        x_cnn = self.cnn(x_cnn)
        x_cnn = x_cnn.view((x_cnn.shape[0], -1))

        x_combined = torch.cat((x_pretrained, x_cnn), dim=1)

        # x = self.dropout(x_combined)
        x = self.linear1(x_combined)
        x = self.relu2(x)
        x = self.linear2(x)

        return x
    

# class ASTPipeline(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.feature_extractor = AutoFeatureExtractor.from_pretrained("bookbot/distil-ast-audioset")
#         # self.feature_extractor = ASTFeatureExtractor()

#     def forward(self, audio):
#         # For some reason it doesn't work on batches so this is a hacky solution
#         pretrained_feat = None
#         print('forward', audio.requires_grad)
#         for sample in audio:
#             sample_feat = self.feature_extractor(sample, sampling_rate=SAMPLE_RATE, return_tensors="pt")['input_values']
#             if pretrained_feat is None:
#                 pretrained_feat = sample_feat
#             else:
#                 pretrained_feat = torch.cat((pretrained_feat, sample_feat), dim=0)
#         return pretrained_feat
    
# class CombinedClassifier(nn.Module):
#     def __init__(self, device):
#         super().__init__()
#         self.device = device

#         self.ast_pipeline = ASTPipeline()
#         self.ast_model = ASTModel.from_pretrained('bookbot/distil-ast-audioset')
#         self.ast_model = self.ast_model.to(self.device)
        
#         for param in self.ast_model.parameters():
#             param.requires_grad = False

#         self.relu1 = nn.ReLU()
#         # self.linear1 = nn.Linear(768+608, 100)
#         self.linear1 = nn.Linear(768, 100)
#         self.relu2 = nn.ReLU()
#         self.linear2 = nn.Linear(100, 10)

#     def forward(self, x):
#         x_pretrained = x.detach()
#         x_pretrained = self.ast_pipeline(x_pretrained)
#         x_pretrained = x_pretrained.to(self.device)
#         x_pretrained = self.ast_model(input_values=x_pretrained).pooler_output
        
#         # x_cnn = self.cnn_pipeline(x, self.sr)
#         # if len(x_cnn.shape) == 3:
#         #     x_cnn = torch.unsqueeze(x_cnn, dim=0)
#         # x_cnn = self.conv(x_cnn)
#         # x_cnn = self.relu(x_cnn)
#         # x_cnn = self.bn(x_cnn)
#         # x_cnn = self.pool(x_cnn)
#         # x_cnn = x_cnn.view((x_cnn.shape[0], -1))

#         # x = torch.cat((x_cnn, x_pretrained), dim=1)
#         x = x_pretrained
#         x = self.relu1(x)
#         x = self.linear1(x)
#         x = self.relu2(x)
#         x = self.linear2(x)
        
#         return x