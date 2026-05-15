import os
import json
import math
from scipy.stats import spearmanr, pearsonr

if __name__ == "__main__":
    models = [ "base-model", "ft-model", "model-diffs" ]
    ablation_comparisons = [ "base-vs-keep", "base-vs-omit", "ft-vs-keep", "ft-vs-omit" ]
    evaluation_metrics = [ "Mean Score Margin", "Min Score Margin", "Categorical Cross Entropy", "NDCG" ]
    granularities = [
        ("layer", "layer"),
        ("window/size2", "window-size2"),
        ("window/size3", "window-size3"),
        ("window/size4", "window-size4"),
        ("window/size6", "window-size6"),
    ]

    # ablation path = "results/ablation-analysis/{granularity[0]}/{ablation_comparison}/{eval_metric}.json"
    # attention_path = "results/{model}/layers/.../{granularity[1]}.json"
    # correlation_path = "results/correlation/{model}/{granularity[1]}/{eval_metric}.json"

    for model in models:
        for granularity in granularities:
            for ablation_comparison in ablation_comparisons:
                os.makedirs(f"results/correlation/{model}/{granularity[1]}/{ablation_comparison}", exist_ok=True)

                for evaluation_metric in evaluation_metrics:
                    spearman_correlation_table = {}
                    pearson_correlation_table = {}
                
                    for feature in os.listdir(f"results/{model}/layers"):
                        ablation_path = f"results/ablation-analysis/{granularity[0]}/{ablation_comparison}/{evaluation_metric}.json"
                        attention_path = f"results/{model}/layers/{feature}/{granularity[1]}.json"

                        with open(ablation_path, "r") as ablation_data_file:
                            with open(attention_path, "r") as attention_data_file:
                                ablation_data = json.load(ablation_data_file)
                                attention_data = json.load(attention_data_file)

                                scorr = spearmanr(ablation_data, attention_data)[0]

                                if not math.isnan(scorr):
                                    spearman_correlation_table[feature] = scorr

                                pcorr = pearsonr(ablation_data, attention_data).correlation

                                if not math.isnan(pcorr):
                                    pearson_correlation_table[feature] = pcorr

                    with open(f"results/correlation/{model}/{granularity[1]}/{ablation_comparison}/{evaluation_metric} - Spearman.json", "w") as correlation_data_file:
                        json.dump(dict(sorted(spearman_correlation_table.items(), key = lambda entry: -abs(entry[1]))), correlation_data_file, indent=4)

                    with open(f"results/correlation/{model}/{granularity[1]}/{ablation_comparison}/{evaluation_metric} - Pearson.json", "w") as correlation_data_file:
                        json.dump(dict(sorted(pearson_correlation_table.items(), key = lambda entry: -abs(entry[1]))), correlation_data_file, indent=4)