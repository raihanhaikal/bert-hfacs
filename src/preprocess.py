import re
import string
from sklearn.model_selection import train_test_split


# Fungsi untuk membersihkan teks
def clean_text(text, remove_stopwords=False):
    text = text.lower()  # Mengubah teks menjadi huruf kecil
    text = re.sub(
        r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE
    )  # Menghapus URL
    text = re.sub(r"\S+@\S+", "", text)  # Menghapus email
    text = re.sub(r"<.*?>", "", text)  # Menghapus tag HTML
    text = re.sub(r"\d+", "", text)  # Menghapus angka
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # Menghapus tanda baca
    text = re.sub(r"[^\x00-\x7f]", r"", text)  # Menghapus karakter khusus
    text = re.sub(r"\s+", " ", text).strip()  # Menghapus spasi berlebih

    if remove_stopwords:
        text = " ".join([word for word in text.split() if word not in stop_words])

    return text


def preprocessed(data):
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

    data = data.rename(columns={"Teks": "text"})
    data = data.rename(columns={"TARGET_LIST": "label"})

    # Define the mapping dictionary for converting the one-hot encoded labels to their respective categories
    label_mapping = {
        "[1.0, 0.0, 0.0, 0.0, 0.0]": "ER",
        "[0.0, 1.0, 0.0, 0.0, 0.0]": "VIO",
        "[0.0, 0.0, 1.0, 0.0, 0.0]": "EF",
        "[0.0, 0.0, 0.0, 1.0, 0.0]": "CO",
        "[0.0, 0.0, 0.0, 0.0, 1.0]": "PF",
    }

    # Convert the TARGET_LIST column to string for mapping
    data["label"] = data["label"].astype(str)

    # Apply the label mapping
    data["label"] = data["label"].map(label_mapping)

    data = data.dropna()
    data.to_csv(
        "E:/code/project-list/bert-hfacs/data/interim/data_preprocessed.csv",
        index=False,
    )

    return data


def split_data(data):
    # train test split
    train_dataset, test_dataset = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=1
    )

    print(f"Train shape: {train_dataset.shape}")
    print(f"Test shape: {test_dataset.shape}")

    # export to csv
    train_dataset.to_csv(
        "E:/code/project-list/bert-hfacs/data/processed/train.csv",
        index=False,
    )
    test_dataset.to_csv(
        "E:/code/project-list/bert-hfacs/data/processed/test.csv",
        index=False,
    )

    return train_dataset, test_dataset
