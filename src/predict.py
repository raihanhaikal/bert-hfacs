import torch
from torch import optim
from dataset import HfacsDataset
from dataloader import HfacsDataloader
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from predict_model import evaluate_model

if __name__ == "__main__":
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
        max_seq_len=512,
        batch_size=32,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
    )

    w2i, i2w = HfacsDataset.LABEL2INDEX, HfacsDataset.INDEX2LABEL

    model = model.cuda()

    evaluate_model(model, test_loader, i2w=i2w)
