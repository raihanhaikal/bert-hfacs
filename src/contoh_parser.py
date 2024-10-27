from argparse import ArgumentParser


def print_opts(opts):
    """Prints the values of all command-line arguments."""
    print("=" * 80)
    print("Opts".center(80))
    print("-" * 80)
    for key in opts.keys():
        if opts[key]:
            print("{:>30}: {:<50}".format(key, opts[key]).center(80))
    print("=" * 80)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, default="exp", help="Experiment name"
    )
    parser.add_argument(
        "--model_dir", type=str, default="save/", help="Model directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="emotion-twitter",
        help="Choose between emotion-twitter, absa-airy, term-extraction-airy, ner-grit, pos-idn, entailment-ui, doc-sentiment-prosa, keyword-extraction-prosa, qa-factoid-itb, news-category-prosa, ner-prosa, pos-prosa",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="bert-base-multilingual-uncased",
        help="Path, url or short name of the model",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Max number of tokens"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--valid_batch_size", type=int, default=4, help="Batch size for validation"
    )
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument(
        "--max_norm", type=float, default=10.0, help="Clipping gradient norm"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--fp16",
        type=str,
        default="",
        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--step_size", type=int, default=1, help="Step size")
    parser.add_argument("--early_stop", type=int, default=3, help="Step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma")
    parser.add_argument("--debug", action="store_true", help="debugging mode")
    parser.add_argument(
        "--force", action="store_true", help="force to rewrite experiment folder"
    )
    parser.add_argument(
        "--no_special_token",
        action="store_true",
        help="not adding special token as the input",
    )
    parser.add_argument("--lower", action="store_true", help="lower case")

    args = vars(parser.parse_args())
    print_opts(args)
    return args


def append_dataset_args(args):
    if args["dataset"] == "emotion-twitter":
        args["task"] = "sequence_classification"
        args["num_labels"] = EmotionDetectionDataset.NUM_LABELS
        args["dataset_class"] = EmotionDetectionDataset
        args["dataloader_class"] = EmotionDetectionDataLoader
        args["forward_fn"] = forward_sequence_classification
        args["metrics_fn"] = emotion_detection_metrics_fn
        args["valid_criterion"] = "F1"
        args["train_set_path"] = "./dataset/emot_emotion-twitter/train_preprocess.csv"
        args["valid_set_path"] = "./dataset/emot_emotion-twitter/valid_preprocess.csv"
        args["test_set_path"] = (
            "./dataset/emot_emotion-twitter/test_preprocess_masked_label.csv"
        )
        args["vocab_path"] = "./dataset/emot_emotion-twitter/vocab_uncased.txt"
        args["embedding_path"] = {
            "fasttext-cc-id-300-no-oov-uncased": "./embeddings/fasttext-cc-id/cc.id.300_no-oov_emotion-twitter_uncased.txt",
            "fasttext-4B-id-300-no-oov-uncased": "./embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_emotion-twitter_uncased.txt",
        }
        args["k_fold"] = 1
        args["word_tokenizer_class"] = TweetTokenizer
    elif args["dataset"] == "absa-airy":
        args["task"] = "multi_label_classification"
        args["num_labels"] = AspectBasedSentimentAnalysisAiryDataset.NUM_LABELS
        args["dataset_class"] = AspectBasedSentimentAnalysisAiryDataset
        args["dataloader_class"] = AspectBasedSentimentAnalysisDataLoader
        args["forward_fn"] = forward_sequence_multi_classification
        args["metrics_fn"] = absa_metrics_fn
        args["valid_criterion"] = "F1"
        args["train_set_path"] = "./dataset/hoasa_absa-airy/train_preprocess.csv"
        args["valid_set_path"] = "./dataset/hoasa_absa-airy/valid_preprocess.csv"
        args["test_set_path"] = (
            "./dataset/hoasa_absa-airy/test_preprocess_masked_label.csv"
        )
        args["vocab_path"] = "./dataset/hoasa_absa-airy/vocab_uncased.txt"
        args["embedding_path"] = {
            "fasttext-cc-id-300-no-oov-uncased": "./embeddings/fasttext-cc-id/cc.id.300_no-oov_absa-airy_uncased.txt",
            "fasttext-4B-id-300-no-oov-uncased": "./embeddings/fasttext-4B-id-uncased/fasttext.4B.id.300.epoch5_uncased_no-oov_absa-airy_uncased.txt",
        }
        args["k_fold"] = 1
        args["word_tokenizer_class"] = TweetTokenizer
    else:
        raise ValueError(f'Unknown dataset name `{args["dataset"]}`')
    return args


def get_eval_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_name", type=str, default="exp", help="Experiment name"
    )
    parser.add_argument(
        "--model_dir", type=str, default="./save", help="Model directory"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="emotion-twitter",
        help="Choose between emotion-twitter, absa-airy, term-extraction-airy, ner-grit, pos-idn, entailment-ui, doc-sentiment-prosa, keyword-extraction-prosa, qa-factoid-itb, news-category-prosa, ner-prosa, pos-prosa",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert-base-multilingual-uncased",
        help="Type of the model",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Max number of tokens"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument("--debug", action="store_true", help="debugging mode")
    parser.add_argument(
        "--no_special_token",
        action="store_true",
        help="not adding special token as the input",
    )
    parser.add_argument("--lower", action="store_true", help="lower case")
    parser.add_argument(
        "--fp16",
        type=str,
        default="",
        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (-1: not distributed)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cuda or cpu)"
    )

    args = vars(parser.parse_args())
    print_opts(args)
    return args