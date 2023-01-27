import json
import process_data
from model import Net, run_model
from model2 import Net as Net2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import torch

LABELS  = [
    'GREETING', 'DATE', 'WEATHER', 'REPEAT', 'YES',
    'CANCEL', 'FLIP_COIN', 'TRANSLATE', 'TIMER', 'DEFINITION',
    'MEANING_OF_LIFE', 'WHAT_CAN_I_ASK_YOU', 'ROLL_DICE', 'MAKE_CALL', 'ALARM'
]

def test(fnc):
    total = 450
    total_correct = 0
    for intent in test_data:
        sentences = test_data[intent]
        labels = [intent] * 30

        pred = fnc(sentences)
        correct = pred.count(intent)
        total_correct += correct
        print(f"number of matched for {intent} : {correct}")

    accuracy = total_correct * 100 / total
    print(f"{total}/{total_correct} : accuracy of {accuracy}")
    return total, total_correct


if __name__ == '__main__':
    test_data = None
    with open('datasets/test_data.json', 'r') as f:
        test_data = json.load(f)


    model = Net()
    print("Running naive model")
    test(lambda sentences: run_model(model, sentences))

    model2 = Net2()

    print("Running fine-tuned model")
    model2.load_state_dict(torch.load("./MODEL"))
    model2.eval()

    def run_model2(sentences):
        labels = []
        Y = model2(sentences)
        for intent_cls in Y:
            labels.append(LABELS[intent_cls.argmax()])
        return labels

    test(run_model2)

