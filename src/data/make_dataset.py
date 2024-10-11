import torch
import ast
from torch.utils.data import DataLoader


class CustomDataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = [
            torch.tensor(ast.literal_eval(label), dtype=torch.long) for label in labels
        ]

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        inputs = self.tokenizer.__call__(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": label,
        }


class CustomDataset2:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = [
            torch.tensor(ast.literal_eval(label), dtype=torch.long) for label in labels
        ]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        # Simplified tokenizer call
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": label,
        }


class CustomDataset3:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts  # This should now be a list of strings
        self.labels = [
            torch.tensor(ast.literal_eval(label), dtype=torch.long) for label in labels
        ]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "labels": label,
        }


def build_dataset(tokenizer_max_len):
    train_dataset = CustomDataset3(
        train.TEXT.tolist(),
        train.TARGET_LIST.values.tolist(),
        tokenizer,
        tokenizer_max_len,
    )
    test_dataset = CustomDataset3(
        test.TEXT.tolist(),
        test.TARGET_LIST.values.tolist(),
        tokenizer,
        tokenizer_max_len,
    )

    return train_dataset, test_dataset


def build_dataloader(train_dataset, test_dataset, batch_size):
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_data_loader, test_data_loader
