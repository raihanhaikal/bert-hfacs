import pandas as pd
from sklearn.model_selection import train_test_split

# Membaca file Excel
data = pd.read_excel(
    "E:/code/project-list/bert-hfacs/data/external-raw/subclass_hfacs_dataset.xlsx",
    sheet_name="Sheet1",
)

# Karena dataset didapatkan dari hasil hfacs manual dengan Pak Ridwan maka di drop kolom "Alasan"
data = data.drop(columns=["Alasan"])

# Mengganti simbol2
data = data.replace("-", 0)
data = data.replace("?", 0)
data = data.replace("--", 0)
data = data.fillna(0)

# Mengganti tipe data

data["ER (LVL1)"] = data["ER (LVL1)"].astype(float)
data["VIO (LVL1)"] = data["VIO (LVL1)"].astype(float)
data["EF (LVL2)"] = data["EF (LVL2)"].astype(float)
data["CO (LVL2)"] = data["CO (LVL2)"].astype(float)
data["PF (LVL2)"] = data["PF (LVL2)"].astype(float)

data["TARGET_LIST"] = data[
    ["ER (LVL1)", "VIO (LVL1)", "EF (LVL2)", "CO (LVL2)", "PF (LVL2)"]
].values.tolist()


data = data.drop(
    columns=["ER (LVL1)", "VIO (LVL1)", "EF (LVL2)", "CO (LVL2)", "PF (LVL2)"]
)

data = data.rename(columns={"Teks": "TEXT"})
data = data.rename(columns={"target_list": "TARGET_LIST"})

# train test split
train_dataset, test_dataset = train_test_split(
    data, test_size=0.2, stratify=data.TARGET_LIST, random_state=1
)

print(f"Train shape: {train_dataset.shape}")
print(f"Test shape: {test_dataset.shape}")

# export to csv
train_dataset.to_csv(
    "/content/drive/MyDrive/Skripsi/Dataset/Subclass/train_dataset_pisah.csv",
    index=False,
)
test_dataset.to_csv(
    "/content/drive/MyDrive/Skripsi/Dataset/Subclass/test_dataset_pisah.csv",
    index=False,
)

####### Augmentasi Chatgpt manual ########


# Membaca file Excel
data = pd.read_excel(
    "E:/code/project-list/bert-hfacs/data/external-raw/subclass_hfacs_dataset_1to1_train_aug_chatgpt.xlsx",
    sheet_name="Sheet1",
)

data.to_csv(
    "E:/code/project-list/bert-hfacs/data/interim/subclass_hfacs_dataset_1to1_train_aug_chatgpt.csv",
    index=False,
)

# Membaca file csv
data = pd.read_csv(
    "E:/code/project-list/bert-hfacs/data/interim/subclass_hfacs_dataset_1to1_train_aug_chatgpt.csv",
    engine="python",
)


"""
value_counts = data["TARGET_LIST"].apply(tuple).value_counts()

class_names = {
    (0.0, 0.0, 0.0, 0.0, 0.0): "Neutral",
    (1.0, 0.0, 0.0, 0.0, 0.0): "ER (LVL1)",
    (0.0, 1.0, 0.0, 0.0, 0.0): "VIO (LVL1)",
    (0.0, 0.0, 1.0, 0.0, 0.0): "EF (LVL2)",
    (0.0, 0.0, 0.0, 1.0, 0.0): "CO (LVL2)",
    (0.0, 0.0, 0.0, 0.0, 1.0): "PF (LVL2)",
}

sorted_value_counts = sorted(value_counts.items(), key=lambda x: class_names[x[0]])

for class_tuple, count in sorted_value_counts:
    class_name = class_names[class_tuple]
    print(f"{class_name} {count}")

"""
