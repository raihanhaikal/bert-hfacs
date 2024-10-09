# Membaca file Excel
data = pd.read_excel(
    "/content/drive/MyDrive/Skripsi/Dataset/Subclass/HFACS Label Full Manual_Subclass_Pisah.xlsx",
    sheet_name="Sheet1",
)

data.to_csv(
    "/content/drive/MyDrive/Skripsi/Dataset/Subclass/dataset_knkt_subclass_pisah.csv",
    index=False,
)

# Membaca file csv
data = pd.read_csv(
    "/content/drive/MyDrive/Skripsi/Dataset/Subclass/dataset_knkt_subclass_pisah.csv",
    engine="python",
)

data.head()

# Membersihkan data

data = data.drop(columns=["Alasan"])
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

# Visualisasi Data

data["TARGET_LIST"] = data[
    ["ER (LVL1)", "VIO (LVL1)", "EF (LVL2)", "CO (LVL2)", "PF (LVL2)"]
].values.tolist()

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
