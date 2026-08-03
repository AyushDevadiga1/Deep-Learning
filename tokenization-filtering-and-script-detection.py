import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Run only once
nltk.download("punkt")
nltk.download("stopwords")

text = "Natural Language Processing is an exciting field of Artificial Intelligence."

print("Original Text:")
print(text)

# Tokenization
tokens = word_tokenize(text)
print("\nTokens:")
print(tokens)

# Filtration
filtered = [
    word for word in tokens
    if word not in string.punctuation
    and word.lower() not in stopwords.words("english")
]

print("\nFiltered Tokens:")
print(filtered)

# Script Validation
print("\nScript Validation:")
print("English (Latin Script)" if text.isascii() else "Non-English Script")