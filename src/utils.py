import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# Forward function for sequence classification
def forward_sequence_classification(
    model, batch_data, i2w, is_test=False, device="cpu", **kwargs
):
    # Unpack batch data
    if len(batch_data) == 3:
        (subword_batch, mask_batch, label_batch) = batch_data
        token_type_batch = None
    elif len(batch_data) == 4:
        (subword_batch, mask_batch, token_type_batch, label_batch) = batch_data

    # Prepare input & label
    subword_batch = torch.LongTensor(subword_batch)
    mask_batch = torch.FloatTensor(mask_batch)
    token_type_batch = (
        torch.LongTensor(token_type_batch) if token_type_batch is not None else None
    )
    label_batch = torch.LongTensor(label_batch)

    if device == "cuda":
        subword_batch = subword_batch.cuda()
        mask_batch = mask_batch.cuda()
        token_type_batch = (
            token_type_batch.cuda() if token_type_batch is not None else None
        )
        label_batch = label_batch.cuda()

    # Forward model
    outputs = model(
        subword_batch,
        attention_mask=mask_batch,
        token_type_ids=token_type_batch,
        labels=label_batch,
    )
    loss, logits = outputs[:2]

    # generate prediction & label list
    list_hyp = []
    list_label = []
    hyp = torch.topk(logits, 1)[1]
    for j in range(len(hyp)):
        list_hyp.append(i2w[hyp[j].item()])
        list_label.append(i2w[label_batch[j][0].item()])

    return loss, list_hyp, list_label


def hfacs_metrics_fn(list_hyp, list_label):
    metrics = {}
    metrics["ACC"] = accuracy_score(list_label, list_hyp)
    metrics["F1"] = f1_score(list_label, list_hyp, average="macro")
    metrics["REC"] = recall_score(list_label, list_hyp, average="macro")
    metrics["PRE"] = precision_score(list_label, list_hyp, average="macro")
    return metrics


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append("{}:{:.2f}".format(key, value))
    return " ".join(string_list)

def save_model(model, save_model_name="model", save_model_dir="E:/code/project-list/bert-hfacs/models/model_trained/"):
    # Pastikan direktori ada, jika tidak buat
    os.makedirs(save_model_dir, exist_ok=True)

    # Gabungkan nama model dengan ekstensi .pth
    save_model_name_with_extension = save_model_name + ".pth"

    # Gabungkan direktori dan nama file model
    save_path = os.path.join(save_model_dir, save_model_name_with_extension)

    # Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    

def load_model(model, load_model_name="model", load_model_dir="E:/code/project-list/bert-hfacs/models/model_trained/"):
    # Pastikan direktori ada, jika tidak buat
    os.makedirs(load_model_dir, exist_ok=True)

    # Gabungkan nama model dengan ekstensi .pth
    load_model_name_with_extension = load_model_name + ".pth"

    # Gabungkan direktori dan nama file model
    load_path = os.path.join(load_model_dir, load_model_name_with_extension)

    # Save model
    model.load_state_dict(
        torch.load(
            load_path,
            weights_only=True,
        )
    )
    print("Model Weight Loaded")
    return model

def plot_metrics(train_losses, train_accuracies):
    """
    Plot training loss and accuracy.

    Parameters:
    - train_losses: List of train losses per epoch
    - train_accuracies: List of train accuracies per epoch
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color='black', linestyle='solid', linewidth=2)
    plt.plot(train_accuracies, label="Train Accuracy", color='black', linestyle='dotted', linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Metric Value")
    plt.title("Training Loss and Accuracy")
    
    # Add grid and ticks
    plt.grid(True)
    plt.tick_params(axis='both', direction='in', length=6)
    
    plt.legend()
    plt.show()
