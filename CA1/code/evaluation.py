from collections import defaultdict
import math
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import ir_datasets


class EvaluationMetrics:
    @staticmethod
    def precision_at_k(retrieved_docs, relevant_docs, k, min_relevant=3):
        """
        retrieved_docs: list of doc_ids
        relevant_docs: dict of {doc_id: relevance_score}
        """
        retrieved_at_k = retrieved_docs[:k]
        relevant_retrieved = [
            d for d in retrieved_at_k if relevant_docs.get(d, 0) >= min_relevant
        ]
        return len(relevant_retrieved) / k if k > 0 else 0

    @staticmethod
    def recall(retrieved_docs, relevant_docs, min_relevant=3):
        relevant_filtered = {
            d for d, score in relevant_docs.items() if score >= min_relevant
        }
        relevant_retrieved = set(retrieved_docs) & relevant_filtered
        return (
            len(relevant_retrieved) / len(relevant_filtered)
            if relevant_filtered
            else 0
        )

    @staticmethod
    def recall_at_k(retrieved_docs, relevant_docs, k, min_relevant=3):
        retrieved_at_k = retrieved_docs[:k]
        relevant_filtered = {
            d for d, score in relevant_docs.items() if score >= min_relevant
        }
        relevant_retrieved = set(retrieved_at_k) & relevant_filtered
        return (
            len(relevant_retrieved) / len(relevant_filtered)
            if relevant_filtered
            else 0
        )

    @staticmethod
    def average_precision(retrieved_docs, relevant_docs, min_relevant=3):
        """
        Computes AP using graded relevance (≥min_relevant threshold).
        """
        relevant_filtered = {
            d for d, score in relevant_docs.items() if score >= min_relevant
        }
        if not relevant_filtered:
            return 0

        relevant_retrieved = 0
        cumulative_precision = 0

        for i, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_filtered:
                relevant_retrieved += 1
                cumulative_precision += relevant_retrieved / i

        return cumulative_precision / len(relevant_filtered)

    @staticmethod
    def mean_average_precision(all_retrieved_docs, all_relevant_docs, min_relevant=3):
        total_ap = 0
        num_queries = len(all_retrieved_docs)
        if num_queries == 0:
            return 0

        for query_id, retrieved_docs in all_retrieved_docs.items():
            relevant_docs = all_relevant_docs.get(query_id, {})
            total_ap += EvaluationMetrics.average_precision(
                retrieved_docs, relevant_docs, min_relevant
            )

        return total_ap / num_queries

    @staticmethod
    def ndcg(retrieved_docs, relevant_docs, k, min_relevant=3):
        """
        Computes normalized DCG with graded relevance.
        """
        if not relevant_docs:
            return 0

        # Compute DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_docs[:k]):
            rel = relevant_docs.get(doc_id, 1)
            # custom dcg: r -> r-1 to reproduce the paper
            dcg += (rel - 1) / math.log2(i + 2)

        # Compute IDCG (Ideal DCG)
        ideal_rels = sorted(
            [score for score in relevant_docs.values()],
            reverse=True,
        )
        # custom idcg: r -> r-1 to reproduce the paper
        idcg = sum(((rel-1)) / math.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))

        return dcg / idcg if idcg > 0 else 0

    @staticmethod
    def mean_reciprocal_rank(all_retrieved_docs, all_relevant_docs, min_relevant=3):
        """
        Computes Mean Reciprocal Rank (MRR) across all queries.
        The reciprocal rank is 1 / rank_of_first_relevant_doc (using min_relevant threshold).
        """
        reciprocal_ranks = []

        for query_id, retrieved_docs in all_retrieved_docs.items():
            relevant_docs = all_relevant_docs.get(query_id, {})
            relevant_filtered = {
                d for d, score in relevant_docs.items() if score >= min_relevant
            }

            rank = 0
            for i, doc_id in enumerate(retrieved_docs, 1):
                if doc_id in relevant_filtered:
                    rank = i
                    break

            reciprocal_ranks.append(1 / rank if rank > 0 else 0)

        return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0


def load_relevance_judgments():
    """
    Load the golden data file and return:
    {query_id: {doc_id: relevance_score}}
    """
    relevance_judgments = defaultdict(dict)
    dataset = ir_datasets.load("antique/test")
    for qrel in dataset.qrels_iter():
        query_id = qrel.query_id
        doc_id = qrel.doc_id
        relevance_score = qrel.relevance
        relevance_judgments[query_id][doc_id] = relevance_score

    return relevance_judgments


def evaluate_scoring_method(
    inverted_index, scoring_method, queries, relevance_judgments, params={}, min_relevant=3
):
    """
    Evaluate a scoring method with graded relevance using multiple metrics.
    Metrics: MAP, MRR, Precision@3, Precision@10, nDCG@10, Recall@3, Recall@10
    """
    all_retrieved_docs = {}
    precision_at_3_scores = []
    precision_at_10_scores = []
    recall_at_3_scores = []
    recall_at_10_scores = []
    ndcg_at_10_scores = []

    for query in queries:
        scores = scoring_method(query, inverted_index, **params)
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        retrieved_docs = [doc_id for doc_id, _ in ranked_docs]
        all_retrieved_docs[query.query_id] = retrieved_docs

        relevant_docs = relevance_judgments.get(query.query_id, {})

        # Precision@3 and Precision@10
        p3 = EvaluationMetrics.precision_at_k(retrieved_docs, relevant_docs, 3, min_relevant)
        p10 = EvaluationMetrics.precision_at_k(retrieved_docs, relevant_docs, 10, min_relevant)
        precision_at_3_scores.append(p3)
        precision_at_10_scores.append(p10)

        # Recall@3 and Recall@10
        r3 = EvaluationMetrics.recall_at_k(retrieved_docs, relevant_docs, 3, min_relevant)
        r10 = EvaluationMetrics.recall_at_k(retrieved_docs, relevant_docs, 10, min_relevant)
        recall_at_3_scores.append(r3)
        recall_at_10_scores.append(r10)

        # nDCG@10
        ndcg10 = EvaluationMetrics.ndcg(retrieved_docs, relevant_docs, 10, min_relevant)
        ndcg_at_10_scores.append(ndcg10)

    # Compute MAP and MRR
    map_score = EvaluationMetrics.mean_average_precision(
        all_retrieved_docs, relevance_judgments, min_relevant
    )
    mrr_score = EvaluationMetrics.mean_reciprocal_rank(
        all_retrieved_docs, relevance_judgments, min_relevant
    )

    # Aggregate average metrics
    results = {
        'MAP': round(float(map_score), 4),
        'MRR': round(float(mrr_score), 4),
        'Precision@3': round(float(np.mean(precision_at_3_scores)), 4),
        'Precision@10': round(float(np.mean(precision_at_10_scores)), 4),
        'Recall@3': round(float(np.mean(recall_at_3_scores)), 4),
        'Recall@10': round(float(np.mean(recall_at_10_scores)), 4),
        'nDCG@10': round(float(np.mean(ndcg_at_10_scores)), 4),
    }

    return results


from scipy.stats import ttest_rel


def compare_methods_vs_baseline(
    methods,  # dict: {method_name: scoring_function}
    params_map,  # dict: {method_name: params_dict}
    baseline_name,  # string: name of baseline method (must be in methods)
    inverted_index,
    queries,
    relevance_judgments,
    min_relevant=3,
):
    """
    Compare multiple retrieval methods against a baseline (paired t-test on per-query AP).

    Args:
        methods: dict mapping method_name -> function(query, inverted_index, **params)
        params_map: dict mapping method_name -> params dict (may be empty dict)
        baseline_name: name of baseline in methods
        inverted_index, queries, relevance_judgments: as used elsewhere
        min_relevant: relevance threshold used by average_precision

    Returns:
        pandas.DataFrame with one row per method and columns:
          ['MAP', 'MRR', 'Precision@3', 'Precision@10', 'Recall@3', 'Recall@10', 'nDCG@10',
           'Better', 'Worse', 'p-val (AP)']
    """
    # 1) compute per-query AP for every method
    # select queries that have relevance judgments (or consider all queries but AP=0 when no relevant)
    query_ids = [q.query_id for q in queries]
    # prepare containers
    per_query_ap = {name: {} for name in methods.keys()}  # method -> {query_id: AP}
    # Also collect retrieved lists for MAP/MRR etc (we can reuse evaluate_scoring_method for aggregated metrics)
    # But for per-query AP we compute AP per query
    for q in queries:
        qid = q.query_id
        relevant_docs = relevance_judgments.get(qid, {})
        for name, func in methods.items():
            params = params_map.get(name, {})
            scores = func(q, inverted_index, **params)
            ranked_docs = [
                doc_id
                for doc_id, _ in sorted(
                    scores.items(), key=lambda x: x[1], reverse=True
                )
            ]
            ap = EvaluationMetrics.average_precision(
                ranked_docs, relevant_docs, min_relevant=min_relevant
            )
            per_query_ap[name][qid] = ap

    # 2) For each method compute aggregate metrics using existing evaluate_scoring_method
    summary_rows = []
    # Build an easy mapping of method->all_retrieved_docs for MAP/MRR (so evaluate_scoring_method could be reused),
    # but simpler: call evaluate_scoring_method directly for each method to get MAP etc.
    for name, func in methods.items():
        params = params_map.get(name, {})
        res = evaluate_scoring_method(
            inverted_index,
            func,
            queries,
            relevance_judgments,
            params=params,
            min_relevant=min_relevant,
        )
        summary_rows.append((name, res))

    # 3) For statistical comparison: paired arrays of AP between method and baseline
    baseline_ap_map = per_query_ap[baseline_name]
    results = []
    for name, res in summary_rows:
        # build paired lists aligned by query order
        aps_baseline = []
        aps_method = []
        for qid in query_ids:
            # Only include queries that appear in relevance_judgments OR include all queries (AP will be 0 if none relevant)
            # here include all queries to keep comparisons consistent
            aps_baseline.append(baseline_ap_map.get(qid, 0.0))
            aps_method.append(per_query_ap[name].get(qid, 0.0))

        aps_baseline = np.array(aps_baseline)
        aps_method = np.array(aps_method)

        # Count better / worse per query
        better = int((aps_method > aps_baseline).sum())
        worse = int((aps_method < aps_baseline).sum())

        # paired t-test on AP values
        try:
            stat, pval = ttest_rel(aps_method, aps_baseline, nan_policy="omit")
            pval = float(pval)
        except Exception:
            stat, pval = float("nan"), float("nan")

        row = {
            "Method": name,
            "MAP": res["MAP"],
            "MRR": res["MRR"],
            "Precision@3": res["Precision@3"],
            "Precision@10": res["Precision@10"],
            "Recall@3": res["Recall@3"],
            "Recall@10": res["Recall@10"],
            "nDCG@10": res["nDCG@10"],
            "Better": better,
            "Worse": worse,
            "p-val (AP)": pval,
        }
        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values(by="MAP", ascending=False).reset_index(drop=True)
    return df
