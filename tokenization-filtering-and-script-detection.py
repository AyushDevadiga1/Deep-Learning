import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# 1. Download ALL required dataset dependencies explicitly
nltk.download('punkt')
nltk.download('punkt_tab')  # Required for modern NLTK tokenization
nltk.download('stopwords')  # Required for filtering and language detection

# Sample text containing English, French, and Spanish
text_sample = "Hello! Running code is fun. Bonjour tout le monde. Hola mi amigo."

print("--- 1. TOKENIZATION & FILTERING ---")
# Tokenize raw text into individual words
all_tokens = word_tokenize(text_sample)

# Strip out English stopwords, punctuation, and standalone numbers
eng_stopwords = set(stopwords.words('english'))
clean_tokens = [
    token for token in all_tokens 
    if token.lower() not in eng_stopwords 
    and token not in string.punctuation 
    and not token.isdigit()
]
print(f"Cleaned Tokens: {clean_tokens}\n")


print("--- 2. SIMPLIFIED SCRIPT / LANGUAGE DETECTION ---")
def quick_detect_language(text):
    tokens = [t.lower() for t in word_tokenize(text)]
    
    # Check text token overlap against NLTK's supported stopword languages
    scores = {}
    for lang in stopwords.fileids():
        lang_words = set(stopwords.words(lang))
        # Count how many words in the text match this specific language's dictionary
        matches = len(set(tokens).intersection(lang_words))
        if matches > 0:
            scores[lang] = matches
            
    # Return the language with highest overlap, or 'Unknown' if no matches found
    return max(scores, key=scores.get).upper() if scores else "UNKNOWN/OTHER"

# Test detection on individual text chunks
chunks = ["Running code is fun", "Bonjour tout le monde", "Hola mi amigo", "XYZ123!!!"]
for chunk in chunks:
    print(f"Text: '{chunk}' -> Detected: {quick_detect_language(chunk)}")
