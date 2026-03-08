from itertools import product
import torch
import tag

class AttentionFeature:
    attending: str # description of what tokens are attending
    attended: str # description of what tokens are being attended
    filters: list # list of tag filters that capture feature pairs

    def __init__(self, attending: str = None, attended: str = None,  filters: list = []):
        self.attending = attending
        self.attended = attended
        self.filters = filters

attention_feature_groups = {
    "type": [
        AttentionFeature(),
        AttentionFeature(attending="Query", attended="Query", filters=[
            tag.filter_first(tag.is_query),
            tag.filter_second(tag.is_query)
        ]),
        AttentionFeature(attending="Document", attended="Query", filters=[
            tag.filter_first(tag.is_document),
            tag.filter_second(tag.is_query)
        ]),
        AttentionFeature(attending="Document", attended="Document", filters=[
            tag.filter_first(tag.is_document),
            tag.filter_second(tag.is_document)
        ])
    ],
    "attending_pos": [
        AttentionFeature(),
        AttentionFeature(attending="Noun", filters=[
            tag.filter_first(tag.is_pos([ "NOUN", "PROPN", "PRON" ]))
        ]),
        AttentionFeature(attending="Verb", filters=[
            tag.filter_first(tag.is_pos([ "VERB", "AUX" ]))
        ]),
        AttentionFeature(attending="Adjective", filters=[
            tag.filter_first(tag.is_pos([ "ADJ", "ADP", "ADV" ]))
        ])
    ],
    "attended_pos": [
        AttentionFeature(),
        AttentionFeature(attended="Noun", filters=[
            tag.filter_second(tag.is_pos([ "NOUN", "PROPN", "PRON" ]))
        ]),
        AttentionFeature(attended="Verb", filters=[
            tag.filter_second(tag.is_pos([ "VERB", "AUX" ]))
        ]),
        AttentionFeature(attended="Adjective", filters=[
            tag.filter_second(tag.is_pos([ "ADJ", "ADP", "ADV" ]))
        ])
    ],
    "attending_rarity": [
        AttentionFeature(),
        AttentionFeature(attending="Common", filters=[
            tag.filter_first(tag.is_word_idf_range([ "very low", "low" ]))
        ]),
        AttentionFeature(attending="Rare", filters=[
            tag.filter_first(tag.is_word_idf_range([ "med", "high", "very high" ]))
        ]),
        AttentionFeature(attending="Very Rare", filters=[
            tag.filter_first(tag.is_word_idf_range([ "high", "very high" ]))
        ])
    ],
    "attended_rarity": [
        AttentionFeature(),
        AttentionFeature(attended="Common", filters=[
            tag.filter_first(tag.is_word_idf_range([ "very low", "low" ]))
        ]),
        AttentionFeature(attended="Rare", filters=[
            tag.filter_first(tag.is_word_idf_range([ "med", "high", "very high" ]))
        ]),
        AttentionFeature(attended="Very Rare", filters=[
            tag.filter_first(tag.is_word_idf_range([ "high", "very high" ]))
        ])
    ],
    "similarity": [
        AttentionFeature(),
        AttentionFeature(attended="Word Identical", filters=[
            tag.filter_combination(tag.are_exact_word_match)
        ]),
        AttentionFeature(attended="Synonymous", filters=[
            tag.filter_combination(tag.are_synonyms)
        ]),
        AttentionFeature(attended="Semantically Related", filters=[
            tag.filter_combination(tag.are_related)
        ])
    ],
    "location": [
        AttentionFeature(),
        AttentionFeature(attended="Neighboring", filters=[
            tag.filter_combination(tag.are_neighbors)
        ])
    ]
}

feature_group_precedence = [ "attending_rarity", "attended_rarity", "location", "similarity", "type", "attending_pos", "attended_pos" ]

def generate_composite_feature_pairs(tagged_tokens: list[tag.TaggedToken]):
    filtered_feature_pairs = []

    for attention_feature_group in feature_group_precedence:
        filtered_feature_pairs.append([
            {
                "attending": attention_feature.attending,
                "attended": attention_feature.attended,
                "pairs": tag.filter_tagged_token_pairs(tagged_tokens, attention_feature.filters)
            }
            for attention_feature in attention_feature_groups[attention_feature_group]
        ])

    composite_features = {}

    for composite_feature in list(product(*filtered_feature_pairs)):
        attending_tokens_description = " ".join([ feature["attending"] for feature in composite_feature if feature["attending"] is not None ])
        attended_tokens_description = " ".join([ feature["attended"] for feature in composite_feature if feature["attended"] is not None ])

        if len(attending_tokens_description) == 0: attending_tokens_description = "All"
        if len(attended_tokens_description) == 0: attended_tokens_description = "All"

        composite_feature_name = f"{attending_tokens_description} Tokens Attending {attended_tokens_description} Tokens"
        
        composite_feature_pairs = set.intersection(*[ feature["pairs"] for feature in composite_feature ])

        composite_features[composite_feature_name] = composite_feature_pairs

    return composite_features

def calculate_attention_mass(attention_matrix: torch.Tensor, pairs: set[tuple[int, int]]):
    raw_feature_attention_mass = 0
    raw_total_attention_mass = 0

    non_sink_feature_attention_mass = 0
    non_sink_total_attention_mass = 0

    for attended_index in range(0, attention_matrix.shape[1]):
        for attending_index in range(attended_index, attention_matrix.shape[0]):
            pair_mass = attention_matrix[attending_index, attended_index].item()

            if (attending_index, attended_index) in pairs: raw_feature_attention_mass += pair_mass

            raw_total_attention_mass += pair_mass 

            if attended_index != 0:
                if (attending_index, attended_index) in pairs: non_sink_feature_attention_mass += pair_mass

                non_sink_total_attention_mass += pair_mass

    unnormalized_attention_mass = raw_feature_attention_mass / raw_total_attention_mass

    normalized_attention_mass = non_sink_feature_attention_mass / non_sink_total_attention_mass
    
    return (unnormalized_attention_mass, normalized_attention_mass)