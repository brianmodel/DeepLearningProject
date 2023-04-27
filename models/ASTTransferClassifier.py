import torch
from torch import nn
from transformers import ASTModel

from transformers import AutoFeatureExtractor
from utils import SAMPLE_RATE

class ASTPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("bookbot/distil-ast-audioset")

    def forward(self, audio):
        # For some reason it doesn't work on batches so this is a hacky solution
        pretrained_feat = None
        for sample in audio:
            sample_feat = self.feature_extractor(sample, sampling_rate=SAMPLE_RATE, return_tensors="pt")['input_values']
            if pretrained_feat is None:
                pretrained_feat = sample_feat
            else:
                pretrained_feat = torch.cat((pretrained_feat, sample_feat), dim=0)
        return pretrained_feat
    
class ASTTransferClassifier(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.ast_pipeline = ASTPipeline()
        self.ast_model = ASTModel.from_pretrained('bookbot/distil-ast-audioset')
        self.ast_model = self.ast_model.to(self.device)
        
        for param in self.ast_model.parameters():
            param.requires_grad = False

        self.relu1 = nn.ReLU()
        # self.linear1 = nn.Linear(768+608, 100)
        self.linear1 = nn.Linear(768, 100)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):
        x_pretrained = self.ast_pipeline(x_pretrained)
        x_pretrained = x_pretrained.to(self.device)
        x_pretrained = self.ast_model(input_values=x_pretrained).pooler_output
        
        x = x_pretrained
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        
        return x