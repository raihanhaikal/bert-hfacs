import torch
import torch.nn as nn
from data.make_dataset import build_dataset, build_dataloader
from models.make_model import ret_model, ret_optimizer, ret_scheduler
from models.train_model import train_fn
from models.predict_model import eval_fn, eval_fn2
from tqdm.notebook import tqdm


def main(config=None):
    config = wandb.config

    train_dataset, valid_dataset = build_dataset(config.tokenizer_max_len)
    train_data_loader, valid_data_loader = build_dataloader(
        train_dataset, valid_dataset, config.batch_size
    )
    print("Length of Train Dataloader: ", len(train_data_loader))
    print("Length of Valid Dataloader: ", len(valid_data_loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_train_steps = int(len(train_dataset) / config.batch_size * 10)

    model = ret_model(n_train_steps, config.dropout)
    optimizer = ret_optimizer(model)
    scheduler = ret_scheduler(optimizer, n_train_steps)
    model.to(device)
    model = nn.DataParallel(model)
    wandb.watch(model)

    n_epochs = config.epochs

    best_val_loss = 100
    for epoch in tqdm(range(n_epochs)):
        train_loss = train_fn(train_data_loader, model, optimizer, device, scheduler)
        eval_loss, preds, labels = eval_fn2(valid_data_loader, model, device)

        jacc_score, recall_score, precision_score = log_metrics2(preds, labels)

        print("JACC score: ", jacc_score)
        print("RECALL score: ", recall_score)
        print("PRECISION score: ", precision_score)

        avg_train_loss, avg_val_loss = (
            train_loss / len(train_data_loader),
            eval_loss / len(valid_data_loader),
        )
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "jacc_score": jacc_score,
                "recall_score": recall_score,
                "precision_score": precision_score,
            }
        )
        print("Average Train loss: ", avg_train_loss)
        print("Average Valid loss: ", avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                "/content/drive/MyDrive/Skripsi/Model BERT/best_model.pt",
            )
            print("Model saved as current val_loss is: ", best_val_loss)
