import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import tag
import attention_features

def attention_heatmap(output_file: str, title: str, tagged_tokens: list[tag.TaggedToken], attention_matrix: torch.Tensor, pairs: set[tuple[int, int]]):
    scores = {}

    for i in range(attention_matrix.shape[0]):
        for j in range(attention_matrix.shape[1]):
            scores[( i, j )] = ((i, j) in pairs, attention_matrix[i, j].item())

    plt.clf()

    fig, ax = plt.subplots()

    x = 0.06

    tile_coords = []

    for tagged_token in tagged_tokens:
        horizontal_text_obj = ax.text(x, 0, tagged_token.text, fontsize=18, va="bottom", ha="left")
        
        ax.text(x - 0.02, x + 0.02, tagged_token.text, fontsize=18, va="bottom", ha="right", rotation=90)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        bbox_pixels = horizontal_text_obj.get_window_extent(renderer=renderer)
        x_pixels = ax.transData.transform([(x, 0)])[0][0]
        pixel_width = bbox_pixels.width
        x_end_pixels = x_pixels + pixel_width
        x_start_data = ax.transData.inverted().transform([(x_pixels, 0)])[0][0]
        x_end_data = ax.transData.inverted().transform([(x_end_pixels, 0)])[0][0]
        width = x_end_data - x_start_data

        tile_coords.append((x, width))

        x += width

    for attended_index in range(len(tagged_tokens)):
        for attending_index in range(attended_index, len(tagged_tokens)):
            if (attending_index, attended_index) in scores:
                is_feature, value = scores[(attending_index, attended_index)]

                color = (0.0, 0.0, value) if attended_index == 0 else (value, 0.0, 0.0) if is_feature else (0.0, value, 0.0)

                ax.add_patch(patches.Rectangle(
                    (tile_coords[attending_index][0], tile_coords[attended_index][0]),
                    tile_coords[attending_index][1],
                    tile_coords[attended_index][1],
                    linewidth=0,
                    facecolor=color
                ))

    unnormalized_attention_mass, normalized_attention_mass = attention_features.calculate_attention_mass(attention_matrix, pairs)

    ax.text(0.06, x - 0.03, title, fontsize=18, fontweight="bold", va="top", ha="left")
    ax.text(
        0.06, x - 0.12,
        f"Unnormalized Feature Attention Mass: {unnormalized_attention_mass:.3f}\nSink Normalized Feature Attention Mass: {normalized_attention_mass:.3f}",
        fontsize=18, va="top", ha="left"
    )

    ax.legend(handles=[
        matplotlib.lines.Line2D([], [], color="none", label="X-Axis: Attending Token"),
        matplotlib.lines.Line2D([], [], color="none", label="Y-Axis: Attended Token"),
        patches.Patch(facecolor="red", label="Feature Attention"),
        patches.Patch(facecolor="green", label="Non-Feature Attention"),
        patches.Patch(facecolor="blue", label="Sink Attention"),
    ], fontsize=18, loc="upper left", bbox_to_anchor=(0.05, x - 0.25), bbox_transform=ax.transData)

    fig.canvas.draw()

    current_fig_width = fig.get_figwidth()
    current_xlim = ax.get_xlim()
    coefficient = current_fig_width / (current_xlim[1] - current_xlim[0])

    fig.set_size_inches(x * coefficient, x * coefficient)

    ax.set_xlim(0, x)
    ax.set_ylim(0, x)
    ax.axis("off")

    plt.savefig(f"{output_file}", dpi=100, bbox_inches="tight")

    plt.close()

def transformer_heatmap(output_file: str, title: str, attention_mass_table: list[list[float]]):
    plt.clf()

    num_attention_layers = len(attention_mass_table)
    num_attention_heads = len(attention_mass_table[0])
    
    data = [[
        attention_mass_table[attention_layer_index][attention_head_index]
        for attention_layer_index in range(num_attention_layers)
    ] for attention_head_index in range(num_attention_heads)]

    black_to_red = mcolors.LinearSegmentedColormap.from_list("black_red", ["black", "red"])

    fig, ax = plt.subplots(figsize=(num_attention_layers * 0.3, num_attention_heads * 0.3))
    
    ax.imshow(data, cmap=black_to_red, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Attention Layer")
    ax.set_ylabel("Attention Head")
    ax.set_xticks(range(num_attention_layers))
    ax.set_yticks(range(num_attention_heads))

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()