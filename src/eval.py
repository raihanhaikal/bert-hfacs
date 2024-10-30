import torch
from dataset import HfacsDataset
from dataloader import HfacsDataloader
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from predict_model import evaluate_model
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

    # Load Tokenizer and Config
    tokenizer = BertTokenizer.from_pretrained(model_path)
    config = BertConfig.from_pretrained(
        model_path,
        num_labels=HfacsDataset.NUM_LABELS,
    )

    # Instantiate model
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        config=config,
    )

    model.load_state_dict(
        torch.load(
            "E:/code/project-list/bert-hfacs/models/model.pth",
            weights_only=True,
        )
    )

    test_dataset_path = "E:/code/project-list/bert-hfacs/data/processed/test.csv"

    test_dataset = HfacsDataset(test_dataset_path, tokenizer, lowercase=True)

    test_loader = HfacsDataloader(
        dataset=test_dataset,
        max_seq_len=args["max_seq_len"],
        batch_size=args["batch_size"],
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    w2i, i2w = HfacsDataset.LABEL2INDEX, HfacsDataset.INDEX2LABEL

    model = model.cuda()

    evaluate_model(model, test_loader, i2w=i2w)
