from typing import Callable
from collections import defaultdict
import string
import spacy
from models import tokenizer
from sentence_transformers import SentenceTransformer, util
import collection_statistics
import math

class TaggedToken:
    id: int
    start: int
    end: int

    categorical_tags: dict[str, str]
    numeric_tags: dict[str, float]
    other_tags: dict

    def __init__(self, index: int, id: int, start: int, end: int, text: str):
        self.index = index
        self.id = id
        self.start = start
        self.end = end
        self.text = text

        self.categorical_tags = defaultdict(lambda: "")
        self.numeric_tags = defaultdict(lambda: 0)
        self.other_tags = {}

    def __str__(self):
        categorical_tags_str = ", ".join(f"{k}=\"{v}\"" for k, v in self.categorical_tags.items())
        numerical_tags_str = ", ".join(f"{k}={v}" for k, v in self.numeric_tags.items())

        tags = ", ".join([ categorical_tags_str, numerical_tags_str ])

        return f"TaggedToken(index={self.index}, text=\"{self.text}\"" + (f", {tags}" if tags else "") + ")"
    
def tokenize(text: str, start_index: int):
    tokens = tokenizer(text, return_offsets_mapping=True)
    
    return [ TaggedToken(index + start_index, id, start, end, text[start : end]) for index, (id, (start, end)) in enumerate(zip(tokens["input_ids"], tokens["offset_mapping"])) ]

pos_tagger = spacy.load("en_core_web_lg", disable=["parser", "ner", "lemmatizer"])

def tag_pos(tagged_tokens: list[TaggedToken], text: str):
    word_list = pos_tagger(text)

    for tagged_token in tagged_tokens:
        if tagged_token.start == tagged_token.end: continue

        # llama tokens often includes space or punctuation, adjust accordingly so its not artificially outside the word bounds
        tagged_token_start = tagged_token.start + (len(tagged_token.text) - len(tagged_token.text.lstrip(string.punctuation + string.whitespace)))
        tagged_token_end = tagged_token.end - (len(tagged_token.text) - len(tagged_token.text.rstrip(string.punctuation + string.whitespace)))

        for word_index, word in enumerate(word_list):
            if tagged_token_start >= word.idx and tagged_token_end <= word.idx + len(word.text):
                tagged_token.categorical_tags["pos"] = word.pos_
                tagged_token.categorical_tags["word"] = word.text
                tagged_token.numeric_tags["word_index"] = word_index

similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

# call after tagging pos to extract word
def tag_embedding(tagged_tokens: list[TaggedToken], text: str):
    embeddings = similarity_model.encode([ tagged_token.categorical_tags["word"].lower().strip() for tagged_token in tagged_tokens ])

    for embedding_index in range(embeddings.shape[0]):
        tagged_tokens[embedding_index].other_tags["embedding"] = embeddings[embedding_index, :]

# call after tagging pos to extract word
def tag_collection_stats(tagged_tokens: list[TaggedToken], text: str):
    for tagged_token in tagged_tokens:
        word = tagged_token.categorical_tags["word"].lower().strip()

        word_idf = math.log(collection_statistics.collection_doc_count / (collection_statistics.word_doc_freq.get(word, 0) + 1))
        word_ido = math.log(collection_statistics.collection_word_count / (collection_statistics.word_occurrences.get(word, 0) + 1))

        token_idf = math.log(collection_statistics.collection_doc_count / (collection_statistics.token_doc_freq.get(tagged_token.id, 0) + 1))
        token_ido = math.log(collection_statistics.collection_token_count / (collection_statistics.token_occurrences.get(tagged_token.id, 0) + 1))

        tagged_token.numeric_tags["word_idf"] = word_idf
        tagged_token.numeric_tags["word_ido"] = word_ido
        tagged_token.numeric_tags["token_idf"] = token_idf
        tagged_token.numeric_tags["token_ido"] = token_ido

        tagged_token.categorical_tags["word_idf_range"] = collection_statistics.idf_range(word_idf)
        tagged_token.categorical_tags["word_ido_range"] = collection_statistics.ido_range(word_ido)
        tagged_token.categorical_tags["token_idf_range"] = collection_statistics.idf_range(token_idf)
        tagged_token.categorical_tags["token_ido_range"] = collection_statistics.ido_range(token_ido)

def tag_query(tagged_tokens: list[TaggedToken], text: str):
    for tagged_token in tagged_tokens:
        tagged_token.categorical_tags["type"] = "query"

def tag_document(tagged_tokens: list[TaggedToken], text: str):
    for tagged_token in tagged_tokens:
        tagged_token.categorical_tags["type"] = "document"

def generate_tagged_tokens(text: str, tags: list[Callable[[list[TaggedToken], str], None]], start_index: int):
    tagged_tokens = tokenize(text, start_index)

    for tag in tags: tag(tagged_tokens, text)

    return tagged_tokens

def is_token(token_id: int):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.id == token_id
    
    return predicate

def is_pos(pos: set[str]):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.categorical_tags["pos"] in pos
    
    return predicate

def is_word_idf_range(idf: set[str]):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.categorical_tags["word_idf_range"] in idf
    
    return predicate

def is_word_ido_range(ido: set[str]):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.categorical_tags["word_ido_range"] in ido
    
    return predicate

def is_token_idf_range(idf: set[str]):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.categorical_tags["token_idf_range"] in idf
    
    return predicate

def is_token_ido_range(ido: set[str]):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.categorical_tags["token_ido_range"] in ido
    
    return predicate

def is_document(tagged_token: TaggedToken):
    return tagged_token.categorical_tags["type"] == "document"

def is_query(tagged_token: TaggedToken):
    return tagged_token.categorical_tags["type"] == "query"

def is_not(predicate: Callable[[TaggedToken], bool]):
    def negated_predicate(tagged_token: TaggedToken):
        return not predicate(tagged_token)
    
    return negated_predicate

def token_satisfies_all(predicates: list[Callable[[TaggedToken], bool]]):
    def conjunctive_predicate(tagged_token: TaggedToken):
        for predicate in predicates:
            if not predicate(tagged_token): return False

        return True
    
    return conjunctive_predicate

def filter_first(predicate: Callable[[TaggedToken], bool]):
    def filter_first_with_predicate(tagged_tokens: list[TaggedToken], pairs: list[tuple[int, int]]):
        return filter(
            lambda pair: predicate(tagged_tokens[pair[0]]),
            pairs
        )
    
    return filter_first_with_predicate

def filter_second(predicate: Callable[[TaggedToken], bool]):
    def filter_second_with_predicate(tagged_tokens: list[TaggedToken], pairs: list[tuple[int, int]]):
        return filter(
            lambda pair: predicate(tagged_tokens[pair[1]]),
            pairs
        )
    
    return filter_second_with_predicate

def are_exact_token_match(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return first_tagged_token.id == second_tagged_token.id

def are_exact_word_match(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return first_tagged_token.categorical_tags["word"].lower().strip() == second_tagged_token.categorical_tags["word"].lower().strip()

def are_synonyms(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return util.cos_sim(
        first_tagged_token.other_tags["embedding"],
        second_tagged_token.other_tags["embedding"]
    ).item() > 0.7

def are_related(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return util.cos_sim(
        first_tagged_token.other_tags["embedding"],
        second_tagged_token.other_tags["embedding"]
    ).item() > 0.3

def are_topical(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return util.cos_sim(
        first_tagged_token.other_tags["embedding"],
        second_tagged_token.other_tags["embedding"]
    ).item() > 0.1

def are_mirror(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return first_tagged_token.index == second_tagged_token.index

def are_adjacent(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return abs(first_tagged_token.index - second_tagged_token.index) == 1

def are_neighbors(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return abs(first_tagged_token.index - second_tagged_token.index) <= 2

def are_same_word_group(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return (
        first_tagged_token.numeric_tags["word_index"] == second_tagged_token.numeric_tags["word_index"] and
        first_tagged_token.categorical_tags["type"] == second_tagged_token.categorical_tags["type"]
    )

def are_not(predicate: Callable[[TaggedToken, TaggedToken], bool]):
    def negated_predicate(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
        return not predicate(first_tagged_token, second_tagged_token)
    
    return negated_predicate

def pair_satisfies_all(predicates: list[Callable[[TaggedToken, TaggedToken], bool]]):
    def conjunctive_predicate(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
        for predicate in predicates:
            if not predicate(first_tagged_token, second_tagged_token): return False

        return True
    
    return conjunctive_predicate

def filter_combination(predicate: Callable[[TaggedToken, TaggedToken], bool]):
    def filter_combination_with_predicate(tagged_tokens: list[TaggedToken], pairs: list[tuple[int, int]]):
        return filter(
            lambda pair: predicate(tagged_tokens[pair[0]], tagged_tokens[pair[1]]),
            pairs
        )
    
    return filter_combination_with_predicate

def filter_tagged_token_pairs(tagged_tokens: list[TaggedToken], pair_filters: list[Callable[[list[TaggedToken], list[tuple[int, int]]], set[tuple[int, int]]]]):
    pairs = []

    for i in range(len(tagged_tokens)):
        for j in range(len(tagged_tokens)):
            pairs.append((i, j))

    for pair_filter in pair_filters:
        pairs = pair_filter(tagged_tokens, pairs)

    return set(pairs)