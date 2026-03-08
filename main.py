import os
from models import get_attention_layers
import tag
from heatmap import heatmap
import attention_features

num_rel_pairs = 500
num_nrel_pairs = 2500

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

    os.makedirs("./results/example-heatmaps", exist_ok=True)

    composite_feature1 = "Noun Tokens Attending Semantically Related Noun Tokens"
    heatmap("results/example-heatmaps/1.png", f"{composite_feature1} (Layer 8, Head 6)", all_tagged_tokens, attention_layers[8][0, 6, :, :], composite_feature_pairs[composite_feature1])

    composite_feature2 = "All Tokens Attending Neighboring Tokens"
    heatmap("results/example-heatmaps/2.png", f"{composite_feature2} (Layer 1, Head 22)", all_tagged_tokens, attention_layers[1][0, 22, :, :], composite_feature_pairs[composite_feature2])

    composite_feature3 = "Rare Tokens Attending Rare Word Identical Tokens"
    heatmap("results/example-heatmaps/3.png", f"{composite_feature3} (Layer 1, Head 18)", all_tagged_tokens, attention_layers[1][0, 18, :, :], composite_feature_pairs[composite_feature3])

    composite_feature4 = "Very Rare Document Tokens Attending Query Tokens"
    heatmap("results/example-heatmaps/4.png", f"{composite_feature4} (Layer 16, Head 17)", all_tagged_tokens, attention_layers[16][0, 17, :, :], composite_feature_pairs[composite_feature4])

if __name__ == "__main__":
    example_heatmap()