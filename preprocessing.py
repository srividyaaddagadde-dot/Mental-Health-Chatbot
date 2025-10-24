import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize once
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def custom_preprocessor(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)
