import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

# Run only once
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

text = "The students are studying Natural Language Processing and playing with computers."

print("Original Text:")
print(text)

tokens = [
    word for word in word_tokenize(text)
    if word not in string.punctuation
]

filtered = [
    word for word in tokens
    if word.lower() not in stopwords.words("english")
]

print("\nAfter Stop Word Removal:")
print(filtered)

stemmer = PorterStemmer()
print("\nAfter Stemming:")
print([stemmer.stem(word) for word in filtered])

lemmatizer = WordNetLemmatizer()
print("\nAfter Lemmatization:")
print([lemmatizer.lemmatize(word) for word in filtered])