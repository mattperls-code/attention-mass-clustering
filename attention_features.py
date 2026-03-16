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
    "attending_type": [
        AttentionFeature(),
        AttentionFeature(attending="Query", filters=[
            tag.filter_first(tag.is_query)
        ]),
        AttentionFeature(attending="Document", filters=[
            tag.filter_first(tag.is_document)
        ])
    ],
    "attended_type": [
        AttentionFeature(),
        AttentionFeature(attended="Query", filters=[
            tag.filter_second(tag.is_query)
        ]),
        AttentionFeature(attended="Document", filters=[
            tag.filter_second(tag.is_document)
        ])
    ],
    # "attending_pos": [
    #     AttentionFeature(),
    #     AttentionFeature(attending="Noun", filters=[
    #         tag.filter_first(tag.is_pos([ "NOUN", "PROPN", "PRON" ]))
    #     ]),
    #     AttentionFeature(attending="Verb", filters=[
    #         tag.filter_first(tag.is_pos([ "VERB", "AUX" ]))
    #     ]),
    #     AttentionFeature(attending="Adjective", filters=[
    #         tag.filter_first(tag.is_pos([ "ADJ" ]))
    #     ]),
    #     AttentionFeature(attending="Adverb", filters=[
    #         tag.filter_first(tag.is_pos([ "ADV" ]))
    #     ])
    # ],
    # "attended_pos": [
    #     AttentionFeature(),
    #     AttentionFeature(attended="Noun", filters=[
    #         tag.filter_second(tag.is_pos([ "NOUN", "PROPN", "PRON" ]))
    #     ]),
    #     AttentionFeature(attended="Verb", filters=[
    #         tag.filter_second(tag.is_pos([ "VERB", "AUX" ]))
    #     ]),
    #     AttentionFeature(attended="Adjective", filters=[
    #         tag.filter_second(tag.is_pos([ "ADJ" ]))
    #     ]),
    #     AttentionFeature(attended="Adverb", filters=[
    #         tag.filter_second(tag.is_pos([ "ADV" ]))
    #     ])
    # ],
    "attending_rarity": [
        AttentionFeature(),
        # AttentionFeature(attending="Common", filters=[
        #     tag.filter_first(tag.is_word_idf_range([ "very low", "low" ]))
        # ]),
        AttentionFeature(attending="Rare", filters=[
            tag.filter_first(tag.is_word_idf_range([ "high", "very high" ]))
        ]),
        AttentionFeature(attending="Very Rare", filters=[
            tag.filter_first(tag.is_word_idf_range([ "very high" ]))
        ])
    ],
    "attended_rarity": [
        AttentionFeature(),
        # AttentionFeature(attended="Common", filters=[
        #     tag.filter_first(tag.is_word_idf_range([ "very low", "low" ]))
        # ]),
        AttentionFeature(attended="Rare", filters=[
            tag.filter_first(tag.is_word_idf_range([ "high", "very high" ]))
        ]),
        AttentionFeature(attended="Very Rare", filters=[
            tag.filter_first(tag.is_word_idf_range([ "very high" ]))
        ])
    ],
    "similarity": [
        AttentionFeature(),
        AttentionFeature(attended="Matching Word", filters=[
            tag.filter_combination(tag.are_exact_word_match)
        ]),
        AttentionFeature(attended="Synonymous", filters=[
            tag.filter_combination(tag.are_synonyms)
        ])
    ],
    "location": [
        AttentionFeature(),
        # AttentionFeature(attended="Adjacent", filters=[
        #     tag.filter_combination(tag.are_adjacent)
        # ]),
        # AttentionFeature(attended="Neighboring", filters=[
        #     tag.filter_combination(tag.are_neighbors)
        # ]),
        # AttentionFeature(attended="Same Word Group", filters=[
        #     tag.filter_combination(tag.are_same_word_group)
        # ]),
        AttentionFeature(attended="Different Word Group", filters=[
            tag.filter_combination(tag.are_not(tag.are_same_word_group))
        ]),
        # to track portion of sunk attention mass
        AttentionFeature(attended="Non-Sink", filters=[
            tag.filter_second(lambda tagged_token: tagged_token.index != 0)
        ])
    ]
}

feature_group_precedence = [ "attending_rarity", "attended_rarity", "location", "similarity", "attending_type", "attended_type" ]

class CompositeFeatureTable:
    def __init__(self, tagged_tokens: list[tag.TaggedToken]):
        self.filtered_feature_pairs = []

        group_index = 0

        for attention_feature_group in feature_group_precedence:
            self.filtered_feature_pairs.append([])

            variant_index = 0

            for attention_feature in attention_feature_groups[attention_feature_group]:
                self.filtered_feature_pairs[-1].append({
                    "group": group_index,
                    "variant": variant_index,
                    "attending": attention_feature.attending,
                    "attended": attention_feature.attended,
                    "pairs": tag.filter_tagged_token_pairs(tagged_tokens, attention_feature.filters)
                })

                variant_index += 1

            group_index += 1

        self.composite_features = {}

        for composite_feature in product(*self.filtered_feature_pairs):
            attending_tokens_description = " ".join([ feature["attending"] for feature in composite_feature if feature["attending"] is not None ])
            attended_tokens_description = " ".join([ feature["attended"] for feature in composite_feature if feature["attended"] is not None ])

            if len(attending_tokens_description) == 0: attending_tokens_description = "All"
            if len(attended_tokens_description) == 0: attended_tokens_description = "All"

            composite_feature_name = f"{attending_tokens_description} Tokens Attending {attended_tokens_description} Tokens"

            self.composite_features[composite_feature_name] = [ (feature["group"], feature["variant"]) for feature in composite_feature ]

    def get(self, key: str):
        return set.intersection(*[ self.filtered_feature_pairs[group_index][variant_index]["pairs"] for group_index, variant_index in self.composite_features[key] ])

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