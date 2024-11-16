from dataset import HfacsDataset
from dataloader import HfacsDataloader
from transformers import BertForSequenceClassification, BertConfig, BertTokenizer
from eval_model import evaluate_model
from parser import get_eval_parser, append_model_args
from utils import load_model, set_seed

if __name__ == "__main__":
    
    print("#################### EVAL BEGIN ####################")
    
    args = get_eval_parser()
    args = append_model_args(args)
    
    # Seed fo CUDA
    set_seed(1)
    
    model_path = args["path"]
    test_dataset_path = "E:/code/project-list/bert-hfacs/data/processed/test.csv"
    
    # Load Tokenizer and Config
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
    
    model = load_model(model, args["load_model_name"])

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
    print("#################### EVAL END ####################")
    
