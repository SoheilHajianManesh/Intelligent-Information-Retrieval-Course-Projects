# query_processing.py
import json
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import ir_datasets


# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class Query:
    def __init__(self, query_id, text):
        self.query_id = query_id
        self.text = text
        self.tokens = []
        self.term_weights = {}

    def preprocess(self, stop_words=None):
        """
        Preprocess the query text:
        - Tokenization
        - Stop-word removal
        - Stemming using Porter Stemmer
        """
        self.tokens = preprocess_text(self.text, stop_words)
        self.term_weights = {term: 1.0 for term in self.tokens}  # Assign weight 1.0 to original terms

    def expand(self, expansion_terms: dict, alpha=1.0, beta=0.3):
        """
        Expand query with controlled weighting.
        alpha: weight for original terms
        beta: weight for expansion terms
        """
        # Normalize expansion weights
        if expansion_terms:
            max_weight = max(expansion_terms.values())
            if max_weight > 0:
                expansion_terms = {t: w/max_weight for t, w in expansion_terms.items()}
        
        # Re-weight original terms
        for t in list(self.term_weights.keys()):
            self.term_weights[t] = alpha
        
        # Add expansion terms
        for t, w in expansion_terms.items():
            if t not in self.term_weights:
                self.term_weights[t] = beta * w
                self.tokens.append(t)
  
def preprocess_text(text, stop_words=None):
    """
    Preprocess the input text and return a list of tokens.
    """
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    
    # Stop-word removal
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

def load_queries():
    """
    Parse the dataset and return a list of Query objects.
    """
    queries = []
    dataset = ir_datasets.load("antique/test")
    for query in dataset.queries_iter():
        query = Query(query.query_id, query.text)
        queries.append(query)
    return queries