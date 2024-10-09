from transformers import AutoTokenizer, AutoModel

bert_model = AutoModel.from_pretrained(
    "indobenchmark/indobert-base-p1",
    cache_dir="E:/code/project-list/bert-hfacs/models",
    clean_up_tokenization_spaces=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    "indobenchmark/indobert-base-p1",
    cache_dir="E:/code/project-list/bert-hfacs/models",
    clean_up_tokenization_spaces=True,
)
