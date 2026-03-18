import os
import json
import heatmap

def calculate_model_diffs():
    base_model_transformer_heatmaps = os.listdir("results/base-model/transformer-heatmaps")
    ft_model_transformer_heatmaps = os.listdir("results/ft-model/transformer-heatmaps")

    mutual_heatmaps = set(base_model_transformer_heatmaps).intersection(set(ft_model_transformer_heatmaps))

    os.makedirs("results/model-diffs/transformer-heatmaps", exist_ok=True)

    for mutual_heatmap in mutual_heatmaps:
        with open(f"results/base-model/transformer-heatmaps/{mutual_heatmap}/normalized.json", "r") as base_model_heatmap_file:
            with open(f"results/ft-model/transformer-heatmaps/{mutual_heatmap}/normalized.json", "r") as ft_model_heatmap_file:
                base_model_heatmap_data = json.load(base_model_heatmap_file)
                ft_model_heatmap_data = json.load(ft_model_heatmap_file)

                diff_heatmap_data = [
                    [
                        ft_model_heatmap_data[layer_index][head_index] - base_model_heatmap_data[layer_index][head_index]
                        for head_index in range(len(base_model_heatmap_data[0]))
                    ]
                    for layer_index in range(len(base_model_heatmap_data))
                ]

                heatmap.transformer_heatmap(
                    f"results/model-diffs/transformer-heatmaps/{mutual_heatmap}.png",
                    f"{mutual_heatmap}\n(Difference Between Normalized Attention Mass in Fine Tuned vs. Base Model)",
                    diff_heatmap_data
                )

if __name__ == "__main__": calculate_model_diffs()