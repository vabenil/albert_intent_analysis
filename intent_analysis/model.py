from transformers import AutoTokenizer, AutoModel
import torch

import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity

# Load model from HuggingFace Hub
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-albert-small-v2')

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = \
            AutoTokenizer\
                .from_pretrained('sentence-transformers/paraphrase-albert-small-v2')

        self.l1 = \
            AutoModel\
                .from_pretrained('sentence-transformers/paraphrase-albert-small-v2')

        self.l2 = mean_pooling

    def forward(self, x):
        x = self.l0(x, padding=True, truncation=True, return_tensors='pt')

        attention_mask = x['attention_mask']

        x = self.l1(**x)
        x = self.l2(x, attention_mask)
        return x


def read_embeddings():
    import pickle
    embeddings = {}
    with open('simple_embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f);
    return embeddings


def run_model(model, sentences):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    model = Net()
    intent_embeddings = read_embeddings()
    labels = list(intent_embeddings.keys())

    Y = []
    pred_embeddings = model(sentences).detach().numpy()
    similarities = cosine_similarity(pred_embeddings, list(intent_embeddings.values()))
    for similairty in similarities:
        Y.append(labels[similairty.argmax()])
    return Y


if __name__ == '__main__':
    # Sentences we want sentence embeddings for
    sentences = ['What time is it?', 'Give me the time']

    model = Net()
    sentence_embeddings = model(sentences).detach()

    print("Sentence embeddings:")
    # Get sentence similairty.
    # Idea is to use this to map sentences to their respective actions
    print(cosine_similarity(sentence_embeddings[0:1], sentence_embeddings[1:]))
    # Would this generalize to simulations as well? I don't know
    # An example of this being utilized for a simple virtual assistant would be:
