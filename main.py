import os
import random
from itertools import islice
from collection_statistics import collection
from models import model, get_attention_layers
import tag
from heatmap import attention_heatmap, transformer_heatmap
import attention_features

num_rel_pairs = 100
num_nrel_pairs = 100

def plot_feature_attention_mass():
    features = [
        "Query Tokens Attending Query Tokens",
        "Document Tokens Attending Query Tokens",
        "Document Tokens Attending Document Tokens",

        "Rare Document Tokens Attending Word Identical Query Tokens",
        "Rare Document Tokens Attending Synonymous Query Tokens",
        "Rare Tokens Attending Different Word Group Word Identical Tokens",
        "Rare Tokens Attending Different Word Group Synonymous Tokens",

        "Rare Document Tokens Attending Rare Word Identical Query Tokens",
        "Rare Document Tokens Attending Rare Synonymous Query Tokens",
        "Rare Tokens Attending Rare Different Word Group Word Identical Tokens",
        "Rare Tokens Attending Rare Different Word Group Synonymous Tokens",

        "Very Rare Document Tokens Attending Very Rare Word Identical Query Tokens",
        "Very Rare Document Tokens Attending Very Rare Synonymous Query Tokens",
        "Very Rare Tokens Attending Very Rare Different Word Group Word Identical Tokens",
        "Very Rare Tokens Attending Very Rare Different Word Group Synonymous Tokens",

        "All Tokens Attending Common Tokens",
        "Common Tokens Attending All Tokens",

        "Noun Tokens Attending Neighboring Verb Tokens",
        "Noun Tokens Attending Neighboring Adjective Tokens",
        "Noun Tokens Attending Neighboring Adverb Tokens",

        "Verb Tokens Attending Neighboring Noun Tokens",
        "Verb Tokens Attending Neighboring Adjective Tokens",
        "Verb Tokens Attending Neighboring Adverb Tokens",

        "Adjective Tokens Attending Neighboring Noun Tokens",
        "Adjective Tokens Attending Neighboring Verb Tokens",
        "Adjective Tokens Attending Neighboring Adverb Tokens",

        "Adverb Tokens Attending Neighboring Noun Tokens",
        "Adverb Tokens Attending Neighboring Verb Tokens",
        "Adverb Tokens Attending Neighboring Adjective Tokens",
    ]

    feature_attention_mass_table = {
        feature_name: {
            "unnormalized": [[ 0.0 ] * model.config.num_attention_heads for _ in range(model.config.num_hidden_layers)],
            "normalized": [[ 0.0 ] * model.config.num_attention_heads for _ in range(model.config.num_hidden_layers)],
        } for feature_name in features
    }

    print("Sampling query-doc pairs")

    documents_store = collection.docs_store()

    queries = { query.query_id: query.text for query in collection.queries_iter() }

    # stored as (query_text, document_id) since docs have efficient retrieval and expensive queries dict gets GC sooner
    query_document_pairs = []

    for qrel in islice(collection.qrels_iter(), num_rel_pairs): query_document_pairs.append((queries[qrel.query_id], qrel.doc_id))
    
    random.seed(3.14159)

    # assume negligible collision rate
    for query in islice(collection.queries_iter(), num_nrel_pairs): query_document_pairs.append((queries[query.query_id], str(random.randint(0, 100000))))
    
    del queries

    print("Finished selecting pairs")

    for pair_index, (query_text, document_id) in enumerate(query_document_pairs, start=1):
        document_text = documents_store.get(document_id).text

        tagged_query_tokens = tag.generate_tagged_tokens(query_text, [
            tag.tag_query,
            tag.tag_pos,
            tag.tag_embedding,
            tag.tag_collection_stats
        ], 0)

        tagged_document_tokens = tag.generate_tagged_tokens(document_text, [
            tag.tag_document,
            tag.tag_pos,
            tag.tag_embedding,
            tag.tag_collection_stats
        ], len(tagged_query_tokens))

        all_tagged_tokens = tagged_query_tokens + tagged_document_tokens

        attention_layers = get_attention_layers(query_text, document_text)

        composite_feature_pairs = attention_features.generate_composite_feature_pairs(all_tagged_tokens)

        for feature in features:
            for attention_layer_index in range(model.config.num_hidden_layers):
                for attention_head_index in range(model.config.num_attention_heads):
                    unnormalized_attention_mass, normalized_attention_mass = attention_features.calculate_attention_mass(
                        attention_layers[attention_layer_index][0, attention_head_index, :, :],
                        composite_feature_pairs[feature]
                    )

                    feature_attention_mass_table[feature]["unnormalized"][attention_layer_index][attention_head_index] += unnormalized_attention_mass
                    feature_attention_mass_table[feature]["normalized"][attention_layer_index][attention_head_index] += normalized_attention_mass

        print(f"Processed pair {pair_index}/{len(query_document_pairs)}")

    for feature in features:
        os.makedirs(f"results/transformer-heatmaps/{feature}", exist_ok=True)
        os.makedirs(f"results/transformer-heatmaps/{feature}", exist_ok=True)

        for attention_layer_index in range(model.config.num_hidden_layers):
            for attention_head_index in range(model.config.num_attention_heads):
                feature_attention_mass_table[feature]["unnormalized"][attention_layer_index][attention_head_index] /= num_rel_pairs + num_nrel_pairs
                feature_attention_mass_table[feature]["normalized"][attention_layer_index][attention_head_index] /= num_rel_pairs + num_nrel_pairs

        transformer_heatmap(
            f"results/transformer-heatmaps/{feature}/unnormalized.png",
            f"{feature}\n(Unnormalized Attention Mass)",
            feature_attention_mass_table[feature]["unnormalized"]
        )

        transformer_heatmap(
            f"results/transformer-heatmaps/{feature}/normalized.png",
            f"{feature}\n(Normalized Attention Mass)",
            feature_attention_mass_table[feature]["normalized"]
        )

def example_heatmap():
    query_text = "query: are house cats the same species as lions?"
    document_text = "document: Lions, or Panthera Leo of the Felidae family, are fearsome predators."

    tagged_query_tokens = tag.generate_tagged_tokens(query_text, [
        tag.tag_query,
        tag.tag_pos,
        tag.tag_embedding,
        tag.tag_collection_stats
    ], 0)

    tagged_document_tokens = tag.generate_tagged_tokens(document_text, [
        tag.tag_document,
        tag.tag_pos,
        tag.tag_embedding,
        tag.tag_collection_stats
    ], len(tagged_query_tokens))

    all_tagged_tokens = tagged_query_tokens + tagged_document_tokens

    attention_layers = get_attention_layers(query_text, document_text)

    composite_feature_pairs = attention_features.generate_composite_feature_pairs(all_tagged_tokens)

    os.makedirs("./results/attention-heatmaps", exist_ok=True)

    composite_feature1 = "Noun Tokens Attending Semantically Related Noun Tokens"
    attention_heatmap("results/attention-heatmaps/1.png", f"{composite_feature1} (Layer 8, Head 6)", all_tagged_tokens, attention_layers[8][0, 6, :, :], composite_feature_pairs[composite_feature1])

    composite_feature2 = "All Tokens Attending Neighboring Tokens"
    attention_heatmap("results/attention-heatmaps/2.png", f"{composite_feature2} (Layer 1, Head 22)", all_tagged_tokens, attention_layers[1][0, 22, :, :], composite_feature_pairs[composite_feature2])

    composite_feature3 = "Rare Tokens Attending Rare Word Identical Tokens"
    attention_heatmap("results/attention-heatmaps/3.png", f"{composite_feature3} (Layer 8, Head 25)", all_tagged_tokens, attention_layers[8][0, 25, :, :], composite_feature_pairs[composite_feature3])

    composite_feature4 = "Very Rare Document Tokens Attending Query Tokens"
    attention_heatmap("results/attention-heatmaps/4.png", f"{composite_feature4} (Layer 16, Head 17)", all_tagged_tokens, attention_layers[16][0, 17, :, :], composite_feature_pairs[composite_feature4])

if __name__ == "__main__":
    plot_feature_attention_mass()

    example_heatmap()