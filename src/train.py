import pandas as pd
from torch import optim
from preprocess import preprocessed, split_data
from dataset import HfacsDataset
from dataloader import HfacsDataloader
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from train_model import train_model
from parser import get_train_parser, append_model_args
from utils import save_model
import os

if __name__ == "__main__":
    args = get_train_parser()
    args = append_model_args(args)

    model_path = args["path"]

    dataset_path = "E:/code/project-list/bert-hfacs/data/processed/train.csv"
    
    # Cek apakah file 'train.csv' ada
    if not os.path.exists(dataset_path):
        print("File 'train.csv' tidak ditemukan, melanjutkan proses membuat dataset")
        
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

    train_model(model, train_loader, optimizer, n_epochs=args["epoch"], i2w=i2w, save_model_name=args["save_model_name"])

    # Save Model
    save_model(model, save_model_name=args["save_model_name"])
