import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class HfacsDataset(Dataset):
    # Static constant variable
    LABEL2INDEX = {"UA": 0, "PRE": 1}
    INDEX2LABEL = {0: "UA", 1: "PRE"}
    NUM_LABELS = 2

    def load_dataset(self, path):
        # Load dataset
        dataset = pd.read_csv(path)
        dataset["label"] = dataset["label"].apply(lambda sen: self.LABEL2INDEX[sen])
        return dataset

    def __init__(
        self, dataset_path, tokenizer, no_special_token=False, *args, **kwargs
    ):
        self.data = self.load_dataset(dataset_path)
        self.tokenizer = tokenizer
        self.no_special_token = no_special_token

    def __getitem__(self, index):
        text, label = self.data.loc[index, "text"], self.data.loc[index, "label"]
        subwords = self.tokenizer.encode(
            text, add_special_tokens=not self.no_special_token
        )
        return np.array(subwords), np.array(label), text

    def __len__(self):
        return len(self.data)
