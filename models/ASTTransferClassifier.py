import torch
from torch import nn
from transformers import ASTModel

class ASTSimpleTransfer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.ast_model = ASTModel.from_pretrained('bookbot/distil-ast-audioset')
        self.ast_model = self.ast_model.to(device)
        
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(768, 100)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(100, 10)
        
    def forward(self, x):
        with torch.no_grad():
          x = self.ast_model(**x).pooler_output
        
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        
        return x