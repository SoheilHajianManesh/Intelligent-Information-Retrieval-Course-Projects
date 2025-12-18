# inverted_index.py
from collections import defaultdict, Counter
import math

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)       # term -> list of (doc_id, freq)
        self.doc_lengths = {}                # doc_id -> length
        self.avg_doc_len = 0
        self.doc_count = 0
        self.term_doc_freq = {}              # term -> number of docs containing term
        self.doc_tokens = {}                 # doc_id -> list of tokens
        self.collection_term_freq = Counter()  # Term frequencies in the entire collection
        self.total_terms = 0                  # Total number of terms in the collection

    def add_document(self, document):
        """
        Add a document to the inverted index.
        """
        doc_id = document.doc_id
        terms = document.tokens
        term_counts = Counter(terms)
        self.doc_lengths[doc_id] = len(terms)
        self.doc_tokens[doc_id] = terms  # Store tokens
        self.doc_count += 1
        self.total_terms += len(terms)

        for term, count in term_counts.items():
            self.index[term].append((doc_id, count))
            self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + 1
            self.collection_term_freq[term] += count  # Update collection term frequencies

    def build_index(self, documents):
        total_length = 0
        for doc in documents:
            doc_id = doc.doc_id
            terms = doc.tokens
            self.doc_tokens[doc_id] = terms  # Store tokens
            term_counts = Counter(terms)
            self.doc_lengths[doc_id] = len(terms)
            total_length += len(terms)
            self.doc_count += 1
            self.total_terms += len(terms)

            for term, count in term_counts.items():
                self.index[term].append((doc_id, count))
                self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + 1
                self.collection_term_freq[term] += count  # Update collection term frequencies

        self.avg_doc_len = total_length / self.doc_count if self.doc_count > 0 else 0

    def get_postings(self, term):
        """
        Retrieve the postings list for a given term.
        """
        return self.index.get(term, [])

    def compute_idf(self, term):
        """
        Compute IDF for a term.
        """
        df = self.term_doc_freq.get(term, 0)
        N = self.doc_count
        if df == 0:
            return 0
        else:
            return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_vocabulary(self):
        """Return the current vocabulary as a list (use set(self.index) for a set)."""
        return list(self.index.keys())

    def compute_collection_prob(self, term):
        """
        Compute the probability of a term in the collection (background model).
        """
        return self.collection_term_freq.get(term, 0) / self.total_terms if self.total_terms > 0 else 0