import os
from transformers import AutoTokenizer, AutoModel
import torch

import torch.nn as nn
import torch.nn.functional as F

# Use AdamW optimizer
from torch.optim import AdamW

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass


LABELS  = [
    'GREETING', 'DATE', 'WEATHER', 'REPEAT', 'YES',
    'CANCEL', 'FLIP_COIN', 'TRANSLATE', 'TIMER', 'DEFINITION',
    'MEANING_OF_LIFE', 'WHAT_CAN_I_ASK_YOU', 'ROLL_DICE', 'MAKE_CALL', 'ALARM'
]

# Load model from HuggingFace Hub
CLASS_NUM = 15
MODEL_NAME = 'sentence-transformers/paraphrase-albert-small-v2'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
albert_model = AutoModel.from_pretrained(MODEL_NAME)

@dataclass
class Config:
    batch_size: int = 16
    shuffle: bool = True
    epochs: int = 3
    seed: int = 22
    lr: float = 0.0021

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = tokenizer
        self.albert = albert_model

        self.l2 = mean_pooling
        self.dropout = nn.Dropout(p=0.1)
        self.l3 = nn.Linear(self.albert.config.hidden_size, CLASS_NUM)

        self.l3.requires_grad = True

        for name, param in self.albert.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.tokenizer(x, padding=True, truncation=True, return_tensors='pt')

        attention_mask = x['attention_mask']

        x = self.albert(**x)
        x = self.l2(x, attention_mask)
        x = F.softmax(self.l3(x), dim=1)
        return x

    def from_tokenized(self, x):
        attention_mask = x['attention_mask']

        x = self.albert(**x)
        x = self.l2(x, attention_mask)
        x = self.dropout(x)
        x = F.softmax(self.l3(x), dim=1)
        return x


def train(model, data_loader, conf=Config(), device='cpu'):
    if type(device) is str:
        device = torch.device(device)

    pid = os.getpid()

    torch.manual_seed(conf.seed)
    optimizer = AdamW(model.parameters(), lr=conf.lr)
    # loss_fn = F.nll_loss
    # loss = nn.L1loss(output, Y)
    loss_fn = nn.CrossEntropyLoss().to(device)

    model.train()
    for epoch in range(conf.epochs):
        count = 0
        for data in data_loader:
            optimizer.zero_grad()

            X = data['review_text']
            Y = data['targets']

            output = model(X)

            loss = loss_fn(output, Y)
            loss.backward()
            optimizer.step()

            if count % 10 == 0:
                print(f"({pid}) "
                      f"Epoch: {epoch}, "
                      f"tdata[{count}] | "
                      f"Loss {round(loss.item(), 6)}")
            count += 1

def run_model(model, sentences):
    labels = []
    Y = model(sentences).detach().numpy()
    for label_tk in Y:
        labels.append(LABELS[label_tk.argmax()])
    return labels

if __name__ == '__main__':
    # Sentences we want sentence embeddings for
    sentences = ['What time is it?', 'Give me the time']

    model = Net()
    Y = model(sentences).detach().numpy()
    print(Y)
