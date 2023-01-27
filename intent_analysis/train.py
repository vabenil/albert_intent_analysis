import torch
from .model2 import Net, train, tokenizer
from .data_loader import create_data_loader, IntentRecognitionDataset
from .process_data import json_to_pandas
import json


if __name__ == '__main__':
    json_data = None;
    with open('datasets/train_data.json', 'r') as json_file:
        json_data = json.load(json_file)

    df = json_to_pandas(json_data);
    df = df.sample(frac=1) # shuffle

    model = Net()
    dataloader = create_data_loader(df, tokenizer, 128, 16)

    train(model, dataloader)
    torch.save(model.state_dict(), "./MODEL")
