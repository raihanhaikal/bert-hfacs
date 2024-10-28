import re
import string
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split

data_train = pd.read_csv("E:/code/project-list/bert-hfacs/data/processed/train.csv")


# Fungsi untuk membersihkan teks
def clean_text(data):
    # Mengubah teks menjadi huruf kecil
    data["text"] = data["text"].str.lower()
    # Menghapus URL
    data["text"] = data["text"].str.replace(r"http\S+|www\S+|https\S+", "", regex=True)
    # Menghapus email
    data["text"] = data["text"].str.replace(r"\S+@\S+", "", regex=True)
    # Menghapus tag HTML
    data["text"] = data["text"].str.replace(r"<.*?>", "", regex=True)
    # Menghapus angka
    data["text"] = data["text"].str.replace(r"\d+", "", regex=True)
    # Menghapus tanda baca
    data["text"] = data["text"].str.replace(
        f"[{re.escape(string.punctuation)}]", "", regex=True
    )
    # Menghapus karakter khusus
    data["text"] = data["text"].str.replace(r"[^\x00-\x7f]", "", regex=True)
    # Menghapus spasi berlebih
    data["text"] = data["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    # Melakukan Stemming/lemmatization
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    data["text"] = data["text"].apply(stemmer.stem)
    # Melakukan Stopword Removal
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    data["text"] = data["text"].apply(stopword_remover.remove)

    return data


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
