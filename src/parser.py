from argparse import ArgumentParser

def get_train_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "indobert_base",
            "indobert_large",
        ],
        required=True,
        help="Nama Model",
    )
    parser.add_argument("--max_seq_len", type=int, default=512, help="max_seq_len")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--lr", type=int, default=1e-4, help="lr")
    parser.add_argument("--epoch", type=int, default=10, help="epoch")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="weight decay untuk optimizer"
    )
    parser.add_argument(
        "--save_model_name",
        type=str,
        default="model",
        help="save nama model",
    )
    args = vars(parser.parse_args())

    return args

def get_eval_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "indobert_base",
            "indobert_large",
        ],
        required=True,
        help="Nama Model",
    )
    parser.add_argument("--max_seq_len", type=int, default=512, help="max_seq_len")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument(
        "--load_model_name",
        type=str,
        default="model",
        help="load nama model",
    )
    
    args = vars(parser.parse_args())

    return args

def append_model_args(args):
    if args["model"] == "indobert_base":
        args["path"] = (
            "E:/code/project-list/bert-hfacs/models/indobert_base"
        )
        args["hidden_layer"] = 12
        args["num_attention_heads"] = 12
        args["hidden_size"] = 768

    elif args["model"] == "indobert_large":
        args["path"] = (
            "E:/code/project-list/bert-hfacs/models/indobert_large"
        )
        args["hidden_layer"] = 24
        args["num_attention_heads"] = 16
        args["hidden_size"] = 1024

    else:
        raise ValueError(f'Unknown model name `{args["model"]}`')
    return args