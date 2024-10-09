# train = pd.read_csv("/content/drive/MyDrive/Skripsi/Dataset/Dataset/Aug After Split/train_dataset_aug.csv", engine="python") # with aug
train = pd.read_csv(
    "/content/drive/MyDrive/Skripsi/Dataset/Subclass/modified_train_dataset_pisah_with_target.csv",
    engine="python",
)  # no aug
# valid = pd.read_csv("/content/drive/MyDrive/Skripsi/Dataset/Dataset/Aug After Split/val_dataset.csv", engine="python")
test = pd.read_csv(
    "/content/drive/MyDrive/Skripsi/Dataset/Subclass/test_dataset_pisah.csv",
    engine="python",
)

train.drop(columns=["Unnamed: 0"], inplace=True)

value_counts_train = train["TARGET_LIST"].value_counts()
value_counts_test = test["TARGET_LIST"].value_counts()


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
