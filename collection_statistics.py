import ir_datasets
from collections import Counter
from itertools import islice
import spacy
from models import tokenizer
import math

collection = ir_datasets.load("msmarco-passage/train")

use_collection_subset = True
subset_collection_size = 10000 # 5000000

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
    doc_words = [ word.text.lower().strip() for word in word_tokenizer(doc.text) ]
    doc_tokens = tokenizer(doc.text).input_ids

    word_doc_freq.update(set(doc_words))
    word_occurrences.update(doc_words)

    token_doc_freq.update(set(doc_tokens))
    token_occurrences.update(doc_tokens)

    collection_word_count += len(doc_words)
    collection_token_count += len(doc_tokens)

print("Finished Collection Statistics Calculation")

# rough bounds for idf and ido ranges

very_low_idf = math.log(100 / 40) # 40-100%
low_idf = math.log(100 / 15) # 15-40%
med_idf = math.log(100 / 4) # 4-15%
high_idf = math.log(100 / 0.5) # 0.5-4%, very high idf < 0.5%

def idf_range(score: float):
    if score < very_low_idf: return "very low"
    if score < low_idf: return "low"
    if score < med_idf: return "med"
    if score < high_idf: return "high"
    else: return "very high"

# uses idf ranges, assuming 25 words per doc
very_low_ido = math.log(50 / 1) # 1 in 50
low_ido = math.log(100 / 1) # 1 in 100
med_ido = math.log(250 / 1) # 1 in 250
high_ido = math.log(2500 / 1) # 1 in 2500, very high ido < 1 in 2500

def ido_range(score: float):
    if score < very_low_ido: return "very low"
    if score < low_ido: return "low"
    if score < med_ido: return "med"
    if score < high_ido: return "high"
    else: return "very high"