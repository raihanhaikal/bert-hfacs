import torch
from tqdm import tqdm
from utils import (
    forward_sequence_classification,
    metrics_to_string,
    hfacs_metrics_fn,
    get_lr,
    plot_metrics_graph,
    plot_metrics_table
)

def train_model(
    model,
    train_loader,
    optimizer,
    n_epochs,
    i2w,
    device="cuda",
    save_model_name=None,
):
    train_losses = []
    train_accuracies = []
    train_f1s = []
    train_recalls = []
    train_precisions = []
    
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

        # Calculate Metrics
        metrics = hfacs_metrics_fn(list_hyp, list_label)
        
        # Append metric list for further plot
        train_losses.append(total_train_loss / len(train_loader))
        train_accuracies.append(metrics["ACC"])
        train_f1s.append(metrics["F1"])
        train_recalls.append(metrics["REC"])
        train_precisions.append(metrics["PRE"])
        
        print(
            "(Epoch {}) TRAIN LOSS:{:.4f} {} LR:{:.8f}".format(
                (epoch + 1),
                total_train_loss / len(train_loader),
                metrics_to_string(metrics),
                get_lr(optimizer),
            )
        )
    
    plot_metrics_graph(train_losses, train_accuracies, file_name=save_model_name)
    plot_metrics_table(train_accuracies, train_f1s, train_recalls, train_precisions, file_name=save_model_name)
    
        
        
