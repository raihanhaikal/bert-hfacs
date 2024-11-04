from argparse import ArgumentParser

def get_train_parser():
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

def get_eval_parser():
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