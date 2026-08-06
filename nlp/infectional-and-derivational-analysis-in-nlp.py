import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Single line to download dependencies
nltk.download(['punkt_tab', 'wordnet', 'omw-1.4'], quiet=True)

print("--- EXECUTING MORPHOLOGICAL ANALYSIS ---")

text = "Students are learning Natural Language Processing using Python."

# Tokenization
words = word_tokenize(text)

# Lemmatizer
lemmatizer = WordNetLemmatizer()

# FIX: Forces both headers to occupy exactly 15 spaces of width
print(f"{'Original Word':<15}{'Root Word':<15}")
print("-" * 30)

for word in words:
    root = lemmatizer.lemmatize(word.lower())
    # FIX: Padding ensures every row stays perfectly aligned regardless of word size
    print(f"{word:<15}{root:<15}")

print("\nWord Generation Example")
root_word = "learn"
print("Root Word:", root_word)

generated_words = [
    root_word,
    root_word + "s",
    root_word + "ed",
    root_word + "ing",
    root_word + "er"
]
print(generated_words)
