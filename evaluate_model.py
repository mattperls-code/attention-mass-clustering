from itertools import islice
import random
from collection_statistics import collection
import reranker

def mean_reciprocal_rank(rel_doc_rankings: list[int]):
    return sum([
        1 / rel_doc_ranking
        for rel_doc_ranking in rel_doc_rankings
    ]) / len(rel_doc_rankings)

# TODO
def normalized_discount_cumulative_gain(rel_doc_rankings: list[int]):
    return 0

num_samples = 10
nrel_docs_per_sample = 10

def evaluate_model(model):
    documents_store = collection.docs_store()
    queries = { query.query_id: query.text for query in collection.queries_iter() }

    with reranker.using_device(model) as reranker_model:
        rel_doc_rankings = []

        for qrel in islice(collection.qrels_iter(), num_samples):
            print("building ranking list")

            query = queries[qrel.query_id]

            rel_doc = qrel.doc_id
            
            scores = [(
                rel_doc,
                reranker.use_model(reranker_model, query, documents_store.get(rel_doc).text)
            )]

            random.seed(rel_doc)

            for _ in range(nrel_docs_per_sample):
                nrel_doc = str(random.randint(0, 100000))

                scores.append((
                    nrel_doc,
                    reranker.use_model(reranker_model, query, documents_store.get(nrel_doc).text)
                ))

            rel_doc_rankings.append(1 + [
                doc == rel_doc
                for doc, _ in sorted(scores, key = lambda doc_score: -doc_score[1])
            ].index(True))

        mrr = mean_reciprocal_rank(rel_doc_rankings)
        ndcg = normalized_discount_cumulative_gain(rel_doc_rankings)

        print(f"MRR: {mrr}, NDCG: {ndcg}")

if __name__ == "__main__":
    evaluate_model(reranker.ft_model)