import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from argparse import ArgumentParser


def stemming(data):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    data["text"] = data["text"].apply(stemmer.stem)
    return data


def stopword_removal(data):
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    data["text"] = data["text"].apply(stopword_remover.remove)
    return data


def make_vectorizer():
    data = pd.read_csv("E:/code/project-list/bert-hfacs/data/processed/train.csv")
    vectorizer = TfidfVectorizer(max_features=5000)
    data = vectorizer.fit_transform(data["text"])
    with open("E:/code/project-list/bert-hfacs/models/ml_model/vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)


def vectorize_data(data):
    with open("E:/code/project-list/bert-hfacs/models/ml_model/vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    data = vectorizer.transform(data["text"])
    return data


def svm_train(text, label):
    svm_model = SVC(kernel="linear", random_state=1)
    svm_model.fit(text, label)
    with open("E:/code/project-list/bert-hfacs/models/ml_model/svm_model.pkl", "wb") as file:
        pickle.dump(svm_model, file)


def nb_train(text, label):
    nb_model = MultinomialNB()
    nb_model.fit(text, label)
    with open("E:/code/project-list/bert-hfacs/models/ml_model/nb_model.pkl", "wb") as file:
        pickle.dump(nb_model, file)


def ml_eval(text, label, model_path):
    # Membuka dan memuat model SVM yang telah disimpan
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Melakukan prediksi pada data teks yang diberikan
    pred = model.predict(text)

    # Menghitung akurasi dan laporan klasifikasi
    accuracy = accuracy_score(label, pred)
    f1 = f1_score(label, pred, average="macro")
    recall = recall_score(label, pred, average="macro")
    precision = precision_score(label, pred, average="macro")
    report = classification_report(label, pred)

    # Menampilkan hasil evaluasi
    print(f"accuracy: {accuracy}")
    print(f"f1: {f1}")
    print(f"recall: {recall}")
    print(f"precision: {precision}")
    print(f"Laporan Klasifikasi:\n{report}")

    return accuracy, report


def get_parser_ml():
    parser = ArgumentParser()
    # Argumen untuk memilih mode (train atau eval)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval"],
        required=True,
        help="'train' untuk pelatihan atau 'eval' untuk evaluasi",
    )
    # Argumen untuk memilih model (svm atau nb)
    parser.add_argument(
        "--model",
        type=str,
        choices=["svm", "nb"],
        required=True,
        help="Pilih model : 'svm' atau 'nb'",
    )

    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    args = get_parser_ml()

    # Memuat data berdasarkan mode
    if args["mode"] == "train":
        ######## TRAIN ########
        data_train = pd.read_csv(
            "E:/code/project-list/bert-hfacs/data/processed/train.csv"
        )

        # Melakukan Preprocess
        stemming(data_train)
        stopword_removal(data_train)
        make_vectorizer()

        # Melakukan vektorisasi terhadap teks train
        data_train_text = vectorize_data(data_train)
        data_train_label = data_train["label"]

        # Melatih model berdasarkan pilihan
        if args["model"] == "svm":
            svm_train(data_train_text, data_train_label)
        elif args["model"] == "nb":
            nb_train(data_train_text, data_train_label)

    elif args["mode"] == "eval":
        ######## EVAL ########
        data_test = pd.read_csv(
            "E:/code/project-list/bert-hfacs/data/processed/test.csv"
        )

        # Melakukan Preprocess
        stemming(data_test)
        stopword_removal(data_test)

        # Melakukan vektorisasi terhadap teks test
        data_test_text = vectorize_data(data_test)
        data_test_label = data_test["label"]

        # Memuat model dan melakukan evaluasi berdasarkan pilihan
        if args["model"] == "svm":
            svm_path = "E:/code/project-list/bert-hfacs/models/ml_model/svm_model.pkl"
            print("Hasil Evaluasi SVM")
            ml_eval(data_test_text, data_test_label, svm_path)
        elif args["model"] == "nb":
            nb_path = "E:/code/project-list/bert-hfacs/models/ml_model/nb_model.pkl"
            print("Hasil Evaluasi Naive Bayes")
            ml_eval(data_test_text, data_test_label, nb_path)



# python ml_classifier.py --mode train --model nb 
# python ml_classifier.py --mode eval --model nb 

# python ml_classifier.py --mode train --model svm
# python ml_classifier.py --mode eval --model svm 
