import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def tokenize(data):
    data["tokenize_text"] = data.cleaned_document.apply(lambda x: list(tokenize(x)))
    data.sample(n=5, random_state=20).iloc[:, -2:]
    return data

from gensim.utils import tokenize
data = pd.read_csv(
    "E:/code/project-list/bert-hfacs/data/processed/train.csv",
)

data = tokenize(data["text"])

data.sample(n=5, random_state=20).iloc[:, -2:]

def stopword_removal(data):


def make_vectorizer():
    data = pd.read_csv("E:/code/project-list/bert-hfacs/data/processed/train.csv")
    nltk.download('stopwords')
    stop_words_indonesia = stopwords.words('indonesian')
    # Transforming text data into TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words=stop_words_indonesia
    )  # Limiting to 5000 features for efficiency
    data = vectorizer.fit_transform(data)
    with open("E:/code/project-list/bert-hfacs/models/vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

make_vectorizer()

def vectorize_data(data):
    with open("E:/code/project-list/bert-hfacs/models/vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)
    data = vectorizer.transform(data)
    return data

def embed_word2vec(data):
    

def embed_fasttext(data):


def svm_train():


def nb_train():
    

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

data_test = data_test["text"]

data_test = vectorizer.transform(data_test)

data_test
# Step 4: Making predictions on the test set
y_pred = svm_model.predict(data_test)

# Step 5: Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

accuracy, report
