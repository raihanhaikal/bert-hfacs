def eval_fn(data_loader, model, device):
    eval_loss = 0.0
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["labels"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            loss = loss_fn(outputs, targets)
            eval_loss += loss.item()
            fin_targets.extend(targets)
            fin_outputs.extend(torch.sigmoid(outputs))
    return eval_loss, fin_outputs, fin_targets


def eval_fn2(data_loader, model, device, threshold=0.5):
    eval_loss = 0.0
    model.eval()
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["labels"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            loss = loss_fn(outputs, targets)
            eval_loss += loss.item()

            # Menggunakan softmax untuk mendapatkan probabilitas
            probs = torch.softmax(outputs, dim=1)
            fin_targets.extend(targets.cpu())

            # Mengubah probabilitas menjadi prediksi biner berdasarkan threshold
            preds = (probs > threshold).float()
            fin_outputs.extend(preds.cpu())

    return eval_loss, fin_outputs, fin_targets
