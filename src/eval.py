import torch
from dataset import HfacsDataset
from dataloader import HfacsDataloader
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from predict_model import evaluate_model
from parser import get_eval_parser, append_model_args


if __name__ == "__main__":
    args = get_eval_parser()
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

    #model.load_state_dict(
    #   torch.load(
    #        "E:/code/project-list/bert-hfacs/models/model.pth",
    #        weights_only=True,
    #    )
    #)
    
    model.load_state_dict(
        torch.load(
            "E:/code/project-list/bert-hfacs/models/model_class.pth",
            weights_only=True,
        )
    )

    test_dataset_path = "E:/code/project-list/bert-hfacs/data/data_class/test_class.csv"

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
