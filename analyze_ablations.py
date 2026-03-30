import math
import os
import json
import reranker
from heatmap import transformer_heatmap
from scipy.stats import spearmanr

def mean_score_margin(rel_score: float, nrel_scores: list[float]):
    return -(rel_score - sum(nrel_scores) / len(nrel_scores))

def min_score_margin(rel_score: float, nrel_scores: list[float]):
    return -(rel_score - max(nrel_scores))

def log_loss(rel_score: float, nrel_scores: list[float]):
    rel_score_exp = math.exp(rel_score)
    nrel_scores_exp = [ math.exp(nrel_score) for nrel_score in nrel_scores ]

    return -math.log(rel_score_exp / (rel_score_exp + sum(nrel_scores_exp)))

def analyze_ablation(ablation_data_path: str):
    with open(ablation_data_path, "r") as ablation_data_file:
        ablation_data = json.load(ablation_data_file)

        mean_score_margins = []
        min_score_margins = []
        log_losses = []

        for logits in ablation_data:
            rel = logits["rel"]
            nrel = logits["nrel"]

            mean_score_margins.append(mean_score_margin(rel, nrel))
            min_score_margins.append(min_score_margin(rel, nrel))
            log_losses.append(log_loss(rel, nrel))

        avg_mean_score_margin = sum(mean_score_margins) / len(mean_score_margins)
        avg_min_score_margin = sum(min_score_margins) / len(min_score_margins)
        avg_log_loss = sum(log_losses) / len(log_losses)
        
        return avg_mean_score_margin, avg_min_score_margin, avg_log_loss
    
def normalize_head_data(head_data: list[list[float]]):
    max_abs = max(abs(value) for layer in head_data for value in layer)

    return [[ value / max_abs for value in layer ] for layer in head_data]

def top_bottom_k(head_data: list[list[float]], k: int):
    flat = [
        (value, layer_index, head_index)
        for layer_index, layer in enumerate(head_data)
        for head_index, value in enumerate(layer)
    ]
    
    sorted_flat = sorted(flat)
    
    return sorted_flat[:k], sorted_flat[-k:]

def top_model_correlations(model_data_name: str, head_data: list[list[float]], k: int):
    corr_table = {}

    flattened_head_data = [ value for layer in head_data for value in layer ]

    for feature in os.listdir(f"results/{model_data_name}/transformer-heatmaps"):
        if os.path.exists(f"results/{model_data_name}/transformer-heatmaps/{feature}/normalized.json"):
            with open(f"results/{model_data_name}/transformer-heatmaps/{feature}/normalized.json", "r") as feature_data_file:
                feature_data = json.load(feature_data_file)

                flattened_feature_data = [ value for layer in feature_data for value in layer ]

                corr, _ = spearmanr(flattened_head_data, flattened_feature_data)

                if not math.isnan(corr): corr_table[feature] = corr

    return sorted(corr_table.items(), key=lambda item: abs(item[1]), reverse=True)[:k]
    
def analyze_ablations():
    avg_mean_score_margin_data = []
    avg_min_score_margin_data = []
    avg_log_loss_data = []

    unablated_avg_mean_score_margin, unablated_avg_min_score_margin, unablated_avg_log_loss = analyze_ablation("results/ablation/none.json")

    for layer_index in range(reranker.ft_model.config.num_hidden_layers):
        avg_mean_score_margin_data.append([])
        avg_min_score_margin_data.append([])
        avg_log_loss_data.append([])

        for head_index in range(reranker.ft_model.config.num_attention_heads):
            ablated_avg_mean_score_margin, ablated_avg_min_score_margin, ablated_avg_log_loss = analyze_ablation(f"results/ablation/layer{layer_index}-head{head_index}.json")

            avg_mean_score_margin_data[-1].append(ablated_avg_mean_score_margin - unablated_avg_mean_score_margin)
            avg_min_score_margin_data[-1].append(ablated_avg_min_score_margin - unablated_avg_min_score_margin)
            avg_log_loss_data[-1].append(ablated_avg_log_loss - unablated_avg_log_loss)

    os.makedirs("results/ablation/analysis", exist_ok=True)

    avg_mean_score_margin_data = normalize_head_data(avg_mean_score_margin_data)
    avg_min_score_margin_data = normalize_head_data(avg_min_score_margin_data)
    avg_log_loss_data = normalize_head_data(avg_log_loss_data)

    transformer_heatmap("results/ablation/analysis/avg_mean_score_margin", "Average Mean Score Margin", avg_mean_score_margin_data)
    transformer_heatmap("results/ablation/analysis/avg_min_score_margin", "Average Min Score Margin", avg_min_score_margin_data)
    transformer_heatmap("results/ablation/analysis/avg_log_loss", "Average Log Loss", avg_log_loss_data)

    model_data_names = [ "ft-model", "base-model" ]
    performance_metrics = {
        "avg_mean_score_margin": avg_mean_score_margin_data,
        "avg_min_score_margin": avg_min_score_margin_data,
        "avg_log_loss": avg_log_loss_data
    }

    for model_data_name in model_data_names:
        for performance_metric, metric_data in performance_metrics.items():
            print()
            print(f"Top corr between {model_data_name} and {performance_metric}")
            print(top_model_correlations(model_data_name, metric_data, 5))

if __name__ == "__main__":
    analyze_ablations()