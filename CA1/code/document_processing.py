# document_processing.py
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import ir_datasets


# Ensure NLTK data is downloaded
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


global_stop_words = set(stopwords.words("english"))


class Document:
    def __init__(self, doc_id, text, metadata=None):
        self.doc_id = doc_id
        self.text = text
        self.tokens = []
        self.metadata = metadata or {}  # Store additional metadata

    def preprocess(self, stop_words=None):
        """
        Preprocess the document text:
        - Tokenization
        - Lowercasing
        - Stop-word removal
        - Stemming
        """
        self.tokens = preprocess_text(self.text, stop_words)


def preprocess_text(text, stop_words=None):
    """
    Preprocess the input text and return a list of tokens.
    Steps:
    - Remove non-alphabetic characters
    - Tokenization
    - Lowercasing
    - Stop-word removal
    - Stemming using Porter Stemmer
    """
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Lowercasing
    tokens = [token.lower() for token in tokens]

    # Stop-word removal
    if stop_words is None:
        stop_words = global_stop_words
    tokens = [token for token in tokens if token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def load_documents():

    dataset = ir_datasets.load("antique")

    try:
        docs_file_path = dataset.docs_path()
    except Exception as e:
        print(
            f"Could not get docs_path(): {e}. Relying on docs_iter() which might fail."
        )
        docs_list = []
        for doc in dataset.docs_iter():
            docs_list.append(Document(doc.doc_id, doc.text))
        return docs_list

    docs_list = []
    print(f"Manually loading documents from: {docs_file_path} with UTF-8 encoding...")

    try:
        with open(docs_file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    doc_id, text = parts
                    docs_list.append(Document(doc_id, text))
                elif len(parts) > 2:
                    doc_id = parts[0]
                    text = "\t".join(parts[1:])
                    docs_list.append(Document(doc_id, text))

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {docs_file_path}")
        print(
            "Please ensure the dataset is downloaded (this might happen on first run)."
        )
        return []  # Return empty list if file not found

    print(f"Successfully loaded {len(docs_list)} documents.")
    return docs_list
