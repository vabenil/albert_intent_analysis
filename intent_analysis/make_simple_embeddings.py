import json
import pickle
import numpy as np

from .model import Net


if __name__ == '__main__':
    json_data = None;
    with open('datasets/train_data.json', 'r') as json_file:
        json_data = json.load(json_file)

    model = Net()
    intent_embeddings = {}
    for intent, sentences in json_data.items():
        sentence_embeddings = np.mean(model(sentences).detach().numpy(), axis=0)
        intent_embeddings[intent] = sentence_embeddings

    with open('simple_embeddings.pickle', 'wb') as embeddings_file:
        pickle.dump(intent_embeddings, embeddings_file)
