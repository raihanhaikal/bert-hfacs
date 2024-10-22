import torch
from tqdm import tqdm
from utils import (
    forward_sequence_classification,
    metrics_to_string,
    hfacs_metrics_fn,
    get_lr,
)


def train_model(
    model,
    train_loader,
    optimizer,
    n_epochs,
    i2w,
    device="cuda",
):
    """
    Fungsi untuk melatih model sequence classification.

    Parameters:
    - model: Model PyTorch yang akan dilatih
    - train_loader: DataLoader untuk data pelatihan
    - optimizer: Optimizer PyTorch
    - n_epochs: Jumlah epoch pelatihan
    - i2w: Indeks ke kata (dictionary) untuk menerjemahkan indeks prediksi
    - hfacs_metrics_fn: Fungsi untuk menghitung metrik pelatihan
    - get_lr: Fungsi untuk mendapatkan nilai learning rate saat ini
    - device: Perangkat yang digunakan, default "cuda"
    """

    for epoch in range(n_epochs):
        model.train()
        torch.set_grad_enabled(True)

        total_train_loss = 0
        list_hyp, list_label = [], []

        train_pbar = tqdm(train_loader, leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            # Forward model
            loss, batch_hyp, batch_label = forward_sequence_classification(
                model, batch_data[:-1], i2w=i2w, device=device
            )

            # Update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss = loss.item()
            total_train_loss += tr_loss

            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label

            train_pbar.set_description(
                "(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format(
                    (epoch + 1), total_train_loss / (i + 1), get_lr(optimizer)
                )
            )

        # Calculate train metric
        metrics = hfacs_metrics_fn(list_hyp, list_label)
        print(
            "(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format(
                (epoch + 1),
                total_train_loss / len(train_loader),
                metrics_to_string(metrics),
                get_lr(optimizer),
            )
        )
