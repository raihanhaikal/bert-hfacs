import pandas as pd
import pickle
import nltk
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


data_train = pd.read_csv("E:/code/project-list/bert-hfacs/data/processed/train.csv")

# Fungsi tokenisasi sederhana menggunakan regex
def tokenize(data):
    data["text"] = data["text"].apply(lambda x: re.findall(r'\b\w+\b', str(x)))
    return data

tokenize(data_train)

def stemming(data):
    # stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    # Menerapkan stemming pada setiap baris teks di kolom 'text'
    data['text'] = data['text'].apply(stemmer.stem)
    return data

stemming(data_train)

def stopword_removal(data):
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    data['text'] = data['text'].apply(stopword_remover.remove)
    return data

stopword_removal(data_train)


def make_vectorizer():
    data = pd.read_csv("E:/code/project-list/bert-hfacs/data/processed/train.csv")
    vectorizer = TfidfVectorizer(
        max_features=5000
    )  # Limiting to 5000 features for efficiency
    data = vectorizer.fit_transform(data)
    with open("E:/code/project-list/bert-hfacs/models/vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

make_vectorizer()

data_train

def vectorize_data(data):
    with open("E:/code/project-list/bert-hfacs/models/vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    x = vectorizer.transform(data['text'])
    return x

data_train_text = vectorize_data(data_train)

dense_array = data_train_text.toarray()

print(dense_array)

def svm_train(data):
    svm_model = SVC(kernel="linear", random_state=1)
    svm_model.fit(data["text"], data["label"])
    with open("svm_model.pkl", "wb") as file:
        pickle.dump(svm_model, file)
        
svm_train(data_train)

def nb_train(data):
    nb_model = MultinomialNB()  # Using a linear kernel
    nb_model.fit(data["text"], data["label"])
    with open("nb_model.pkl", "wb") as file:
        pickle.dump(svm_model, file)
        
nb_train(data_train)

data_test = pd.read_csv("E:/code/project-list/bert-hfacs/data/processed/test.csv")

def ml_eval():

    


data = vectorize_data(data_train["text"])

data

data = pd.read_csv(
    "E:/code/project-list/bert-hfacs/data/interim/data_preprocessed.csv",
)

data.head()

# Step 1: Preprocessing and Feature Extraction
# Splitting the data into features (X) and target (y)
X = data["text"]
y = data["label"]

X.head()
y.head()
# Transforming text data into TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=5000
)  # Limiting to 5000 features for efficiency
X_tfidf = vectorizer.fit_transform(X)

X_tfidf

# Step 2: Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=1
)


# Step 3: Training the SVM classifier
svm_model = SVC(kernel="linear", random_state=1)  # Using a linear kernel
svm_model.fit(X_train, y_train)

data_test = pd.read_csv(
    "E:/code/project-list/bert-hfacs/data/processed/test.csv",
)



data_test = vectorizer.transform(data_test)

data_test
# Step 4: Making predictions on the test set
y_pred = svm_model.predict(data_test)

# Step 5: Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

accuracy, report
