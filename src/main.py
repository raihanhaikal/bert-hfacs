import pandas as pd
import torch
from torch import optim
from preprocess import preprocessed, split_data
from dataset import HfacsDataset
from dataloader import HfacsDataloader
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from train_model import train_model
from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "indobert_base_lite",
            "indobert_base",
            "indobert_large_lite",
            "indobert_large",
        ],
        required=True,
        help="Nama Model",
    )
    parser.add_argument("--max_seq_len", type=int, default=512, help="max_seq_len")
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--lr", type=int, default=5e-6, help="lr")
    parser.add_argument("--epoch", type=int, default=5, help="epoch")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="weight decay untuk optimizer"
    )
    parser.add_argument(
        "--save_model_name",
        type=str,
        default="model.pth",
        help="save nama model, tambahkan .pth",
    )
    args = vars(parser.parse_args())

    return args


def append_model_args(args):
    if args["model"] == "indobert_base_lite":
        args["path"] = (
            "E:/code/project-list/bert-hfacs/models/models--indobenchmark--indobert-base-p1/snapshots/c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"
        )
        args["hidden_layer"] = 12
        args["num_attention_heads"] = 12
        args["hidden_size"] = 768

    elif args["model"] == "indobert_base":
        args["path"] = (
            "E:/code/project-list/bert-hfacs/models/models--indobenchmark--indobert-base-p1/snapshots/c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"
        )
        args["hidden_layer"] = 12
        args["num_attention_heads"] = 12
        args["hidden_size"] = 768

    elif args["model"] == "indobert_large_lite":
        args["path"] = (
            "E:/code/project-list/bert-hfacs/models/models--indobenchmark--indobert-base-p1/snapshots/c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"
        )
        args["hidden_layer"] = 24
        args["num_attention_heads"] = 16
        args["hidden_size"] = 1024

    elif args["model"] == "indobert_large":
        args["path"] = (
            "E:/code/project-list/bert-hfacs/models/models--indobenchmark--indobert-base-p1/snapshots/c2cd0b51ddce6580eb35263b39b0a1e5fb0a39e2"
        )
        args["hidden_layer"] = 24
        args["num_attention_heads"] = 16
        args["hidden_size"] = 1024

    else:
        raise ValueError(f'Unknown model name `{args["model"]}`')
    return args


if __name__ == "__main__":
    args = get_parser()
    args = append_model_args(args)

    model_path = args["path"]
    # Membaca file Excel
    data_raw = pd.read_excel(
        "E:/code/project-list/bert-hfacs/data/raw/subclass_hfacs_dataset.xlsx",
        sheet_name="Sheet1",
    )

    data_preprocessed = preprocessed(data_raw)
    train_dataset, test_dataset = split_data(data_preprocessed)

    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(
        model_path,
        num_hidden_layers=args["hidden_layer"],
        num_attention_heads=args["num_attention_heads"],
        hidden_size=args["hidden_size"],
        num_labels=HfacsDataset.NUM_LABELS,
    )

    # Instantiate model
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        config=config,
    )

    train_dataset_path = "E:/code/project-list/bert-hfacs/data/processed/train.csv"

    train_dataset = HfacsDataset(train_dataset_path, tokenizer, lowercase=True)

    train_loader = HfacsDataloader(
        dataset=train_dataset,
        max_seq_len=args["max_seq_len"],
        batch_size=args["batch_size"],
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )

    w2i, i2w = HfacsDataset.LABEL2INDEX, HfacsDataset.INDEX2LABEL

    optimizer = optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )
    model = model.cuda()

    train_model(model, train_loader, optimizer, n_epochs=args["epoch"], i2w=i2w)

    # Save Model
    torch.save(model.state_dict(), args["save_model_name"])
