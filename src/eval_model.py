import torch
import pandas as pd
from tqdm import tqdm
from utils import (
    forward_sequence_classification,
    metrics_to_string,
    hfacs_metrics_fn,
)

def evaluate_model(model, test_loader, i2w, device="cuda", output_file="pred.txt"):
    model.eval()
    
    # Freeze Layer
    torch.set_grad_enabled(False)

    list_hyp, list_label = [], []

    pbar = tqdm(test_loader, leave=True, total=len(test_loader))
    for i, batch_data in enumerate(pbar):
        _, batch_hyp, batch_label = forward_sequence_classification(
            model, batch_data[:-1], i2w=i2w, device=device
        )
        list_hyp += batch_hyp
        list_label += batch_label

    # Hitung metrik evaluasi
    metrics = hfacs_metrics_fn(list_hyp, list_label)
    print("{}".format(metrics_to_string(metrics)))

    # Simpan prediksi ke file CSV
    df = pd.DataFrame({"label": list_hyp}).reset_index()
    df.to_csv(output_file, index=False)

    print(df)

    return metrics, df
