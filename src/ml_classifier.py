import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


"""
def tokenize(data):
    data["text"] = data["text"].apply(lambda x: re.findall(r"\b\w+\b", str(x)))
    return data


tokenize(data_train)
"""


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
    with open("E:/code/project-list/bert-hfacs/models/vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)


def vectorize_data(data):
    with open("E:/code/project-list/bert-hfacs/models/vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    data = vectorizer.transform(data["text"])
    return data


def svm_train(text, label):
    svm_model = SVC(kernel="linear", random_state=1)
    svm_model.fit(text, label)
    with open("E:/code/project-list/bert-hfacs/models/svm_model.pkl", "wb") as file:
        pickle.dump(svm_model, file)


def nb_train(text, label):
    nb_model = MultinomialNB()
    nb_model.fit(text, label)
    with open("E:/code/project-list/bert-hfacs/models/nb_model.pkl", "wb") as file:
        pickle.dump(nb_model, file)


def ml_eval(text, label, model_path):
    # Membuka dan memuat model SVM yang telah disimpan
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Melakukan prediksi pada data teks yang diberikan
    pred = model.predict(text)

    # Menghitung akurasi dan laporan klasifikasi
    accuracy = accuracy_score(label, pred)
    report = classification_report(label, pred)

    # Menampilkan hasil evaluasi
    print(f"Akurasi: {accuracy}")
    print(f"Laporan Klasifikasi:\n{report}")

    return accuracy, report


######## TRAIN ########
data_train = pd.read_csv("E:/code/project-list/bert-hfacs/data/processed/train.csv")

# Melakukan Preprocess
stemming(data_train)
stopword_removal(data_train)
make_vectorizer()

# Melakukan vektorisasi terhadap teks train
data_train_text = vectorize_data(data_train)
data_train_label = data_train["label"]

# Melakukan train untuk membuat model svm dan nb
svm_train(data_train_text, data_train_label)
nb_train(data_train_text, data_train_label)

######## EVAL ########
data_test = pd.read_csv("E:/code/project-list/bert-hfacs/data/processed/test.csv")

stemming(data_test)
stopword_removal(data_test)

data_test_text = vectorize_data(data_test)
data_test_label = data_test["label"]


svm_path = "E:/code/project-list/bert-hfacs/models/svm_model.pkl"
nb_path = "E:/code/project-list/bert-hfacs/models/nb_model.pkl"

ml_eval(data_test_text, data_test_label, svm_path)
ml_eval(data_test_text, data_test_label, nb_path)
