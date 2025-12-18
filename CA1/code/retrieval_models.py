# retrieval_models.py
from collections import defaultdict
import math


def bm25_score(query, inverted_index, k=1.5, b=0.75):
    """
    Compute BM25 scores for a given query against all documents in the inverted index.

    Parameters:
    - query: Query object
    - inverted_index: InvertedIndex object
    - k: BM25 k parameter (default=1.5)
    - b: BM25 b parameter (default=0.75)

    Returns:
    - scores: Dictionary mapping doc_id to BM25 score
    """
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq * (k + 1)
            denominator = freq + k * (1 - b + b * (doc_len / avg_doc_len))
            score = idf * (numerator / denominator)
            scores[doc_id] += query_weight * score
    return scores


def first_suggested_approach(query, inverted_index):
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        for doc_id, _ in postings:
            scores[doc_id] += idf  # count once per document
    return scores


def second_suggested_approach(query, inverted_index, k=1.5):
    scores = defaultdict(float)
    for term in query.tokens:
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            numerator = freq * (k + 1)
            denominator = freq + k
            score = numerator / denominator
            scores[doc_id] += score
    return scores


def third_suggested_approach(query, inverted_index, k=1.5):
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq * (k + 1)
            denominator = freq + k * ((doc_len / avg_doc_len))
            score = idf * (numerator / denominator)
            scores[doc_id] += query_weight * score
    return scores


def fourth_suggested_approach(query, inverted_index):
    scores = defaultdict(float)
    for term in query.tokens:
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            scores[doc_id] += query_weight
    return scores


def fifth_suggested_approach(query, inverted_index, k=1.5, b=0.75):
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq
            denominator = freq + k * (1 - b + b * (doc_len / avg_doc_len))
            score = (idf**2) * (numerator / denominator)
            scores[doc_id] += query_weight * score
    return scores


def sixth_suggested_approach(query, inverted_index, delta, k=1.5, b=0.75):
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq * (k + 1)
            denominator = freq + k * (1 - b + b * (doc_len / avg_doc_len))
            score = idf * ((numerator / denominator) + delta)
            scores[doc_id] += query_weight * score
    return scores


# BM25S
def seventh_suggested_approach(query, inverted_index, alpha, k=1.5, b=0.75):
    scores = defaultdict(float)
    for term in query.tokens:
        idf = inverted_index.compute_idf(term)
        postings = inverted_index.get_postings(term)
        query_weight = query.term_weights.get(term, 1.0)
        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len
            numerator = freq * (k + 1)
            denominator = (
                freq + k * (1 - b + b * (doc_len / avg_doc_len)) + (alpha * freq)
            )
            score = idf * (numerator / denominator)
            scores[doc_id] += query_weight * score
    return scores


def pivoted_length_normalization(query, inverted_index, b=0.75):
    scores = defaultdict(float)
    M = inverted_index.doc_count

    for term in query.tokens:
        postings = inverted_index.get_postings(term)
        df = len(postings)
        if df == 0:
            continue

        idf = math.log((M + 1) / df)

        query_weight = query.term_weights.get(term, 1.0)

        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len

            numerator = math.log(1 + math.log(1 + freq))

            denominator = 1 - b + b * (doc_len / avg_doc_len)

            score = query_weight * freq * (numerator / denominator) * idf
            scores[doc_id] += score

    return scores


def pivoted_length_normalization_v2(query, inverted_index, b=0.75):
    scores = defaultdict(float)
    M = inverted_index.doc_count

    for term in query.tokens:
        postings = inverted_index.get_postings(term)
        df = len(postings)
        if df == 0:
            continue

        idf = math.log((M + 1) / df)

        query_weight = query.term_weights.get(term, 1.0)

        for doc_id, freq in postings:
            doc_len = inverted_index.doc_lengths[doc_id]
            avg_doc_len = inverted_index.avg_doc_len

            numerator = math.log(1 + freq)

            denominator = 1 - b + b * (doc_len / avg_doc_len)

            score = query_weight * freq * (numerator / denominator) * idf
            scores[doc_id] += score

    return scores
