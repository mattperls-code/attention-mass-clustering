import os
from itertools import islice
import random
import json
from collection_statistics import collection
import reranker

num_samples = 100
nrel_docs_per_sample = 9

def evaluate_model(reranker_model, output_path: str):
    documents_store = collection.docs_store()
    queries = { query.query_id: query.text for query in collection.queries_iter() }

    logit_groups = []

    for qrel in islice(collection.qrels_iter(), num_samples):
        random.seed(qrel.query_id)

        logits = reranker.use_model(
            reranker_model,
            queries[qrel.query_id],
            [
                documents_store.get(qrel.doc_id).text,
                *[
                    documents_store.get(str(random.randint(0, 1000000))).text
                    for _ in range(nrel_docs_per_sample)
                ]
            ]
        )

        logit_groups.append({
            "rel": logits[0],
            "nrel": logits[1:]
        })

    with open(output_path, "w") as output_file:
        json.dump(logit_groups, output_file, indent=4)

if __name__ == "__main__":
    os.makedirs("results/ablation", exist_ok=True)

    with reranker.using_device(reranker.ft_model):
        print("Evaluating unablated")
        evaluate_model(reranker.ft_model, "results/ablation/none.json")

        # for layer_index in range(reranker.ft_model.config.num_hidden_layers):
        #     for head_index in range(reranker.ft_model.config.num_attention_heads):
        #         print(f"Ablating layer {layer_index}, head {head_index}")

        #         with reranker.use_lora_ablated_model([ (layer_index, head_index) ]) as ablated_model:
        #             evaluate_model(ablated_model, f"results/ablation/layer{layer_index}-head{head_index}.json")

        for layer_index in range(reranker.ft_model.config.num_hidden_layers):
            print(f"Ablating layer {layer_index}")

            with reranker.use_lora_ablated_model([
                (layer_index, head_index)
                for head_index in range(reranker.ft_model.config.num_attention_heads)
            ]) as ablated_model:
                evaluate_model(ablated_model, f"results/ablation/layer{layer_index}.json")

        for layer_index in range(reranker.ft_model.config.num_hidden_layers):
            print(f"Ablating layer {layer_index}")

            with reranker.use_lora_ablated_model([
                (other_layer_index, head_index)
                for head_index in range(reranker.ft_model.config.num_attention_heads)
                for other_layer_index in range(reranker.ft_model.config.num_hidden_layers)
                if other_layer_index != layer_index
            ]) as ablated_model:
                evaluate_model(ablated_model, f"results/ablation/keep-layer{layer_index}.json")