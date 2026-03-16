import ir_datasets
from collections import Counter
from itertools import islice
import spacy
import reranker
import math

collection = ir_datasets.load("msmarco-passage/train")

use_collection_subset = True
subset_collection_size = 500000

collection_doc_count = subset_collection_size if use_collection_subset else collection.docs_count()
collection_word_count = 0
collection_token_count = 0

# doc freq is number of docs the term appears in, occurrences is total count of the term
word_doc_freq = Counter()
word_occurrences = Counter()
token_doc_freq = Counter()
token_occurrences = Counter()

# process chosen subset

doc_iter = islice(collection.docs_iter(), subset_collection_size) if use_collection_subset else collection.docs_iter()

print("Starting Collection Statistics Calculation")

word_tokenizer = spacy.blank("en")

for doc in doc_iter:
    doc_words = [ word.text.lower() for word in word_tokenizer(doc.text) ]
    doc_tokens = reranker.tokenizer(doc.text).input_ids

    word_doc_freq.update(set(doc_words))
    word_occurrences.update(doc_words)

    token_doc_freq.update(set(doc_tokens))
    token_occurrences.update(doc_tokens)

    collection_word_count += len(doc_words)
    collection_token_count += len(doc_tokens)

print("Finished Collection Statistics Calculation")

# rough bounds for idf and ido ranges

very_low_idf = math.log(1000 / 200) # 200 in every thousand docs
low_idf = math.log(1000 / 100) # 100 in every thousand docs
med_idf = math.log(1000 / 25) # 25 in every thousand docs
high_idf = math.log(1000 / 2) # 2 in every thousand docs

def idf_range(score: float):
    if score < very_low_idf: return "very low"
    if score < low_idf: return "low"
    if score < med_idf: return "med"
    if score < high_idf: return "high"
    else: return "very high"

# uses idf ranges, assuming 25 words per doc
very_low_ido = math.log(25 * 1000 / 200) # 200 in every 25 thousand words
low_ido = math.log(25 * 1000 / 100) # 100 in every 25 thousand words
med_ido = math.log(25 * 1000 / 25) # 25 in every 25 thousand words
high_ido = math.log(25 * 1000 / 2) # 2 in every 25 thousand words

def ido_range(score: float):
    if score < very_low_ido: return "very low"
    if score < low_ido: return "low"
    if score < med_ido: return "med"
    if score < high_ido: return "high"
    else: return "very high"