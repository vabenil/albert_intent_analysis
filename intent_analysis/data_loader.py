from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torch

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = IntentRecognitionDataset(
        reviews=df.prompt.to_numpy(),
        targets=df.intent.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    print(ds.reviews.shape)
    print(ds.targets.shape)
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


# TODO: remove max length
class IntentRecognitionDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    # def my_collate_fn(data):
    #     print(f"data: {data}")
    #     return tuple(data)

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }
