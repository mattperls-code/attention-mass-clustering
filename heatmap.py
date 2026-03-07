import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tag
from models import get_attention_layers

def heatmap(output_file: str, tagged_tokens: list[tag.TaggedToken], attention_matrix: torch.Tensor, pairs: set[tuple[int, int]]):
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

if __name__ == "__main__":
    query_text = "query: are house cats the same species as lions?"
    document_text = "document: Lions, or Panthera Leo of the Felidae family, are fearsome predators."

    query_tokens = tag.generate_tagged_tokens(query_text, [
        tag.tag_query,
        tag.tag_pos,
        tag.tag_embedding,
        tag.tag_collection_stats
    ], 0)

    document_tokens = tag.generate_tagged_tokens(document_text, [
        tag.tag_document,
        tag.tag_pos,
        tag.tag_embedding,
        tag.tag_collection_stats
    ], len(query_tokens))

    all_tokens = query_tokens + document_tokens

    focus_pairs = tag.filter_tagged_token_pairs(all_tokens, [
        tag.filter_first(tag.token_satisfies_all([
            tag.is_document,
            tag.is_pos([ "NOUN", "PROPN" ])
        ])),
        tag.filter_second(tag.token_satisfies_all([
            tag.is_query,
            tag.is_pos([ "NOUN", "PROPN" ])
        ])),
        tag.filter_combination(tag.are_related)
    ])

    attention_layers = [ attention_layer.detach().cpu() for attention_layer in get_attention_layers(query_text, document_text) ]

    heatmap("demo.png", all_tokens, attention_layers[8][0, 6, :, :], focus_pairs) # (11, 17), (11, 19), (11, 26), (8, 6), (16, 17)
    
    # for attention_layer_index, attention_layer in enumerate(attention_layers):
    #     os.makedirs(f"results/example-heatmaps/layer{attention_layer_index}", exist_ok=True)

    #     for attention_head_index in range(attention_layer.shape[1]):
    #         attention_head = attention_layer[0, attention_head_index, :, :]

    #         heatmap(f"results/example-heatmaps/layer{attention_layer_index}/head{attention_head_index}.png", all_tokens, attention_head, focus_pairs)

    # prevent bus error caused by similarity model's internal destructor
    os._exit(0)