import pandas as pd
import torch
from torch import optim
from preprocess import preprocessed, split_data
from dataset import HfacsDataset
from dataloader import HfacsDataloader
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from train_model import train_model


if __name__ == "__main__":
    # Membaca file Excel
    data_raw = pd.read_excel(
        "E:/code/project-list/bert-hfacs/data/raw/subclass_hfacs_dataset.xlsx",
        sheet_name="Sheet1",
    )

    data_preprocessed = preprocessed(data_raw)
    train_dataset, test_dataset = split_data(data_preprocessed)

    # Load Tokenizer and Config
    tokenizer = BertTokenizer.from_pretrained(
        "E:/code/project-list/bert-hfacs/models/models--indobenchmark--indobert-base-p1/snapshots/c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"
    )
    config = BertConfig.from_pretrained(
        "E:/code/project-list/bert-hfacs/models/models--indobenchmark--indobert-base-p1/snapshots/c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"
    )
    config.num_labels = HfacsDataset.NUM_LABELS

    # Instantiate model
    model = BertForSequenceClassification.from_pretrained(
        "E:/code/project-list/bert-hfacs/models/models--indobenchmark--indobert-base-p1/snapshots/c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2",
        config=config,
    )

    train_dataset_path = "E:/code/project-list/bert-hfacs/data/processed/train.csv"

    train_dataset = HfacsDataset(train_dataset_path, tokenizer, lowercase=True)

    train_loader = HfacsDataloader(
        dataset=train_dataset,
        max_seq_len=512,
        batch_size=32,
        num_workers=2,
        shuffle=True,
        pin_memory=True,
    )

    w2i, i2w = HfacsDataset.LABEL2INDEX, HfacsDataset.INDEX2LABEL

    optimizer = optim.Adam(model.parameters(), lr=5e-6)
    model = model.cuda()

    train_model(model, train_loader, optimizer, n_epochs=5, i2w=i2w)

    # Save Model
    torch.save(model.state_dict(), "model.pth")

    # Save Model dengan optimizer
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_with_optimizer.pth",
    )
