import torch
from torch import optim
from dataset import HfacsDataset
from dataloader import HfacsDataloader
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from train_model import train_model
from parser import get_train_parser, append_model_args
from utils import save_model, set_seed

if __name__ == "__main__":
    
    print("#################### TRAIN BEGIN ####################")
    args = get_train_parser()
    args = append_model_args(args)
    
    # Seed for CUDA
    set_seed(1)

    model_path = args["path"]

    train_dataset_path = "E:/code/project-list/bert-hfacs/data/processed/train.csv"

    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    train_dataset = HfacsDataset(train_dataset_path, tokenizer, lowercase=True)
    
    train_loader = HfacsDataloader(
        dataset=train_dataset,
        max_seq_len=args["max_seq_len"],
        batch_size=args["batch_size"],
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    
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
    
    w2i, i2w = HfacsDataset.LABEL2INDEX, HfacsDataset.INDEX2LABEL
    
    optimizer = optim.Adam(
        model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_model(model, train_loader, optimizer, n_epochs=args["epoch"], i2w=i2w, save_model_name=args["save_model_name"])

    # Save Model
    save_model(model, save_model_name=args["save_model_name"])
    
    print("#################### TRAIN END ####################")
