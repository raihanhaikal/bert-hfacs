import torch
from tqdm import tqdm
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#import matplotlib.pyplot as plt
from utils import (
    forward_sequence_classification,
    metrics_to_string,
    hfacs_metrics_fn,
    plot_metrics_table_eval,
    save_pred_to_txt
)

def evaluate_model(model, test_loader, i2w, device="cuda", load_model_name = None):

    list_hyp, list_label = [], []
    eval_accuracies = []
    eval_f1s = []
    eval_recalls = []
    eval_precisions = []
    
    model.eval()
    
    # Freeze Layer
    torch.set_grad_enabled(False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pbar = tqdm(test_loader, leave=True, total=len(test_loader))
    for i, batch_data in enumerate(pbar):
        _, batch_hyp, batch_label = forward_sequence_classification(
            model, batch_data[:-1], i2w=i2w, device=device
        )
        list_hyp += batch_hyp
        list_label += batch_label

    # Hitung metrik evaluasi
    #cm = confusion_matrix(list_label, list_hyp, labels=["UA", "PRE"])
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["UA", "PRE"])
    #disp.plot()
    #plt.show()
    
    metrics = hfacs_metrics_fn(list_hyp, list_label)
    
    eval_accuracies.append(metrics["ACC"])
    eval_f1s.append(metrics["F1"])
    eval_recalls.append(metrics["REC"])
    eval_precisions.append(metrics["PRE"])    
    print("{}".format(metrics_to_string(metrics)))
    
    plot_metrics_table_eval(eval_accuracies, eval_f1s, eval_recalls, eval_precisions, file_name=load_model_name)
    
    df = save_pred_to_txt(list_hyp, file_name=load_model_name)
    
    print(df)
    
    return metrics
