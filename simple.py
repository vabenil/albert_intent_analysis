import json;
import process_data
from model import Net
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def read_embeddings():
    import pickle
    embeddings = {}
    with open('simple_embeddings.pickle', 'rb') as f:
        embeddings = pickle.load(f);
    return embeddings


def run_model(model, sentences):
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
    model = Net()
    sentences = ["what does anachronistic mean", "How is the weather?"]
    print(run_model(model, sentences))
