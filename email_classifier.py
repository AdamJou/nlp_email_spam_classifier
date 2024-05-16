import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Funkcje pomocnicze do przetwarzania tekstu
def download_nltk_packages():
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)

def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^\\w\\s]', '', text)
    text = ' '.join(text.split())
    return text

def preprocess_text(texts, lemmatizer, stop_words):
    cleaned_texts = [clean_text(text) for text in texts]
    tokens = [word_tokenize(text) for text in cleaned_texts]
    lemmas = [[lemmatizer.lemmatize(word) for word in token if word not in stop_words] for token in tokens]
    filtered_texts = [' '.join(lemma) for lemma in lemmas]
    return filtered_texts

# Funkcja do trenowania modelu i zwracania metryk
def train_and_evaluate():
    download_nltk_packages()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    data = pd.read_csv('spam_NLP.csv')
    data['processed'] = preprocess_text(data['MESSAGE'].tolist(), lemmatizer, stop_words)
    
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(data['processed'])
    labels = data['CATEGORY']  # Załóżmy, że etykiety są już numeryczne 0 (not spam) i 1 (spam)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1),  # pos_label jest teraz 1 dla spamu
        "recall": recall_score(y_test, y_pred, pos_label=1),
        "f1_score": f1_score(y_test, y_pred, pos_label=1),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    return classifier, vectorizer, metrics

    # Pobieranie i przygotowywanie danych
    download_nltk_packages()
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    data = pd.read_csv('spam_NLP.csv')
    data['processed'] = preprocess_text(data['MESSAGE'].tolist(), lemmatizer, stop_words)
    
    # Wektoryzacja tekstu
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(data['processed'])
    labels = data['CATEGORY']

    # Podział na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Trenowanie modelu
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Ocena modelu
    y_pred = classifier.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label='MESSAGE'),
        "recall": recall_score(y_test, y_pred, pos_label='MESSAGE'),
        "f1_score": f1_score(y_test, y_pred, pos_label='MESSAGE'),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()  # Konwersja na listę dla łatwości przetwarzania w HTML
    }

    return classifier, vectorizer, metrics

# Przechowywanie modelu i wektoryzatora w pamięci
classifier, vectorizer, model_metrics = train_and_evaluate()

# Funkcja do klasyfikacji pojedynczego emaila
def classify_email(email_text):
    prepared_text = preprocess_text([email_text], WordNetLemmatizer(), set(stopwords.words('english')))
    email_features = vectorizer.transform(prepared_text)
    prediction = classifier.predict(email_features)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'
