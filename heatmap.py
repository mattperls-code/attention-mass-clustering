from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tag

def heatmap(output_file: str, tagged_tokens: list[tag.TaggedToken], scores: dict[tuple[int, int], tuple[bool, float]]):
    plt.clf()

    fig, ax = plt.subplots()

    x = 0.06

    tile_coords = []

    for tagged_token in tagged_tokens:
        horizontal_text_obj = ax.text(x, 0, tagged_token.text, fontsize=18, va="bottom", ha="left")
        
        ax.text(0, x, tagged_token.text, fontsize=18, va="bottom", ha="left", rotation=90)

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

    for first_token_index in range(len(tagged_tokens)):
        for second_token_index in range(len(tagged_tokens)):
            is_feature, value = scores[(first_token_index, second_token_index)]

            color = (value, 0.0, 0.0) if is_feature else (0.0, value, 0.0)

            ax.add_patch(patches.Rectangle(
                (tile_coords[first_token_index][0], tile_coords[second_token_index][0]),
                tile_coords[first_token_index][1],
                tile_coords[second_token_index][1],
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