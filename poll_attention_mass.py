import os
import json
import random
from itertools import islice
from collection_statistics import collection
import torch
import reranker
import tag
from heatmap import transformer_heatmap
import attention_features

num_rel_pairs = 1
num_nrel_pairs = 0

total_pairs = num_rel_pairs + num_nrel_pairs

if __name__ == "__main__":
    features = [
        "All Tokens Attending Non-Sink Tokens",
        "Query Tokens Attending Non-Sink Tokens",
        "Document Tokens Attending Non-Sink Tokens",



        "Query Tokens Attending Query Tokens",
        "Document Tokens Attending Document Tokens",
        "Document Tokens Attending Query Tokens",



        "All Tokens Attending Rare Tokens",
        "All Tokens Attending Very Rare Tokens",

        "Rare Tokens Attending Rare Tokens",
        "Rare Tokens Attending Very Rare Tokens",

        "Very Rare Tokens Attending Rare Tokens",
        "Very Rare Tokens Attending Very Rare Tokens",

        "Rare Tokens Attending All Tokens",
        "Very Rare Tokens Attending All Tokens",



        "All Tokens Attending Rare Different Word Group Synonymous Tokens",
        "All Tokens Attending Very Rare Different Word Group Synonymous Tokens",

        "Rare Tokens Attending Rare Different Word Group Synonymous Tokens",
        "Rare Tokens Attending Very Rare Different Word Group Synonymous Tokens",

        "Very Rare Tokens Attending Rare Different Word Group Synonymous Tokens",
        "Very Rare Tokens Attending Very Rare Different Word Group Synonymous Tokens",

        "Rare Tokens Attending Different Word Group Synonymous Tokens",
        "Very Rare Tokens Attending Different Word Group Synonymous Tokens",



        "All Tokens Attending Rare Different Word Group Matching Word Tokens",
        "All Tokens Attending Very Rare Different Word Group Matching Word Tokens",

        "Rare Tokens Attending Rare Different Word Group Matching Word Tokens",
        "Rare Tokens Attending Very Rare Different Word Group Matching Word Tokens",

        "Very Rare Tokens Attending Rare Different Word Group Matching Word Tokens",
        "Very Rare Tokens Attending Very Rare Different Word Group Matching Word Tokens",

        "Rare Tokens Attending Different Word Group Matching Word Tokens",
        "Very Rare Tokens Attending Different Word Group Matching Word Tokens",




        "Document Tokens Attending Rare Query Tokens",
        "Document Tokens Attending Very Rare Query Tokens",

        "Rare Document Tokens Attending Rare Query Tokens",
        "Rare Document Tokens Attending Very Rare Query Tokens",

        "Very Rare Document Tokens Attending Rare Query Tokens",
        "Very Rare Document Tokens Attending Very Rare Query Tokens",

        "Rare Document Tokens Attending Query Tokens",
        "Very Rare Document Tokens Attending Query Tokens",



        "Document Tokens Attending Rare Different Word Group Synonymous Query Tokens",
        "Document Tokens Attending Very Rare Different Word Group Synonymous Query Tokens",

        "Rare Document Tokens Attending Rare Different Word Group Synonymous Query Tokens",
        "Rare Document Tokens Attending Very Rare Different Word Group Synonymous Query Tokens",

        "Very Rare Document Tokens Attending Rare Different Word Group Synonymous Query Tokens",
        "Very Rare Document Tokens Attending Very Rare Different Word Group Synonymous Query Tokens",

        "Rare Document Tokens Attending Different Word Group Synonymous Query Tokens",
        "Very Rare Document Tokens Attending Different Word Group Synonymous Query Tokens",



        "Document Tokens Attending Rare Different Word Group Matching Word Query Tokens",
        "Document Tokens Attending Very Rare Different Word Group Matching Word Query Tokens",

        "Rare Document Tokens Attending Rare Different Word Group Matching Word Query Tokens",
        "Rare Document Tokens Attending Very Rare Different Word Group Matching Word Query Tokens",

        "Very Rare Document Tokens Attending Rare Different Word Group Matching Word Query Tokens",
        "Very Rare Document Tokens Attending Very Rare Different Word Group Matching Word Query Tokens",

        "Rare Document Tokens Attending Different Word Group Matching Word Query Tokens",
        "Very Rare Document Tokens Attending Different Word Group Matching Word Query Tokens"
    ]

    num_layers = reranker.model.config.num_hidden_layers
    num_heads = reranker.model.config.num_attention_heads

    feature_attention_mass_table = {
        feature: {
            "unnormalized": torch.zeros(num_layers, num_heads),
            "normalized": torch.zeros(num_layers, num_heads)
        } for feature in features
    }

    print("Sampling query-doc pairs")

    documents_store = collection.docs_store()
    queries = { query.query_id: query.text for query in collection.queries_iter() }

    query_document_pairs = []
    for qrel in islice(collection.qrels_iter(), num_rel_pairs):
        query_document_pairs.append((queries[qrel.query_id], qrel.doc_id))

    random.seed(3.14159)
    for query in islice(collection.queries_iter(), num_nrel_pairs):
        query_document_pairs.append((queries[query.query_id], str(random.randint(0, 100000))))

    del queries

    print("Finished selecting pairs\n")

    for pair_index, (raw_query_text, document_id) in enumerate(query_document_pairs, start=1):
        query_text = "query: " + raw_query_text
        document_text = "document: " + documents_store.get(document_id).text

        attention_layers = reranker.get_attention_layers(query_text, document_text).cpu()

        tagged_query_tokens = tag.generate_tagged_tokens(query_text, [
            tag.tag_query,
            tag.tag_pos,
            tag.tag_stopword,
            tag.tag_embedding,
            tag.tag_collection_stats
        ], 0)

        tagged_document_tokens = tag.generate_tagged_tokens(document_text, [
            tag.tag_document,
            tag.tag_pos,
            tag.tag_stopword,
            tag.tag_embedding,
            tag.tag_collection_stats
        ], len(tagged_query_tokens))

        all_tagged_tokens = tagged_query_tokens + tagged_document_tokens
        composite_feature_table = attention_features.CompositeFeatureTable(all_tagged_tokens)
        
        seq_len = attention_layers.shape[-1]

        for feature in features:
            feature_mask = composite_feature_table.get_mask(feature, seq_len)
            unnorm, norm = attention_features.calculate_attention_mass_batched(attention_layers, feature_mask)

            feature_attention_mass_table[feature]["unnormalized"] += unnorm
            feature_attention_mass_table[feature]["normalized"] += norm

        print("Finished polling composite features")
        print(f"Processed pair {pair_index}/{len(query_document_pairs)}\n")

    for feature in features:
        unnorm = feature_attention_mass_table[feature]["unnormalized"] / total_pairs
        norm = feature_attention_mass_table[feature]["normalized"] / total_pairs

        os.makedirs(f"results/transformer-heatmaps/{feature}", exist_ok=True)

        unnorm_list = unnorm.tolist()
        norm_list = norm.tolist()

        with open(f"results/transformer-heatmaps/{feature}/unnormalized.json", "w") as f:
            json.dump(unnorm_list, f, indent=4)

        with open(f"results/transformer-heatmaps/{feature}/normalized.json", "w") as f:
            json.dump(norm_list, f, indent=4)

        transformer_heatmap(
            f"results/transformer-heatmaps/{feature}/unnormalized.png",
            f"{feature}\n(Unnormalized Attention Mass)",
            unnorm_list
        )

        transformer_heatmap(
            f"results/transformer-heatmaps/{feature}/normalized.png",
            f"{feature}\n(Normalized Attention Mass)",
            norm_list
        )