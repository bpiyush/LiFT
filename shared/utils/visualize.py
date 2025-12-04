"""Helpers for visualization"""
import os
import itertools
from os.path import exists

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import PIL
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm

# try:
#     import librosa
#     import librosa.display
# except:
#     exit("Failed to import librosa. Please install.")


try:
    from IPython.display import Audio, Markdown, display
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    Audio = Markdown = display = None

try:
    from ipywidgets import Button, HBox, VBox, Text, Label, HTML, widgets
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False
    # Create dummy classes for type hints
    Button = HBox = VBox = Text = Label = HTML = widgets = None

from shared.utils.log import tqdm_iterator

import warnings
warnings.filterwarnings("ignore")

# try:
#     import torchvideotransforms
# except:
#     print("Failed to import torchvideotransforms. Proceeding without.")
#     print("Please install using:")
#     print("pip install git+https://github.com/hassony2/torch_videovision")


# define predominanat colors
COLORS = {
    "pink": (242, 116, 223),
    "cyan": (46, 242, 203),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
}


def get_predominant_color(color_key, mode="RGB", alpha=0):
    assert color_key in COLORS.keys(), f"Unknown color key: {color_key}"
    if mode == "RGB":
        return COLORS[color_key]
    elif mode == "RGBA":
        return COLORS[color_key] + (alpha,)


def show_single_image(image: np.ndarray, figsize: tuple = (8, 8), title: str = None, cmap: str = None, ticks=False):
    """Show a single image."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if isinstance(image, Image.Image):
        image = np.asarray(image)

    ax.set_title(title)
    ax.imshow(image, cmap=cmap)
    
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def show_grid_of_images(
        images: np.ndarray,
        n_cols: int = 4,
        figsize: tuple = (8, 8),
        subtitlesize=14,
        cmap=None,
        subtitles=None,
        title=None,
        save=False,
        savepath="sample.png",
        titlesize=20,
        ysuptitle=0.8,
        xlabels=None,
        sizealpha=0.7,
        show=True,
        row_labels=None,
        aspect=None,
        width_ratios=None,
        return_as_pil=False,
    ):
    """Show a grid of images."""
    n_cols = min(n_cols, len(images))

    copy_of_images = images.copy()
    for i, image in enumerate(copy_of_images):
        if isinstance(image, Image.Image):
            image = np.asarray(image)
            copy_of_images[i] = image

    if subtitles is None:
        subtitles = [None] * len(images)

    if xlabels is None:
        xlabels = [None] * len(images)
    
    if row_labels is None:
        num_rows = int(np.ceil(len(images) / n_cols))
        row_labels = [None] * num_rows

    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, width_ratios=width_ratios,
    )
    if len(images) == 1:
        axes = np.array([[axes]])
    for i, ax in enumerate(axes.flat):
        if i < len(copy_of_images):
            if len(copy_of_images[i].shape) == 2 and cmap is None:
                cmap="gray"
            ax.imshow(copy_of_images[i], cmap=cmap, aspect=aspect)
            ax.set_title(subtitles[i], fontsize=subtitlesize)
        ax.set_xlabel(xlabels[i], fontsize=sizealpha * subtitlesize)
        ax.set_xticks([])
        ax.set_yticks([])

        col_idx = i % n_cols
        if col_idx == 0:
            ax.set_ylabel(row_labels[i // n_cols], fontsize=sizealpha * subtitlesize)

    fig.tight_layout()
    plt.tight_layout()
    plt.suptitle(title, y=ysuptitle, fontsize=titlesize)

    if return_as_pil:
        fig.canvas.draw()
        pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        return pil_image

    # print(f"Saving to {savepath}")
    if save:
        plt.savefig(savepath, bbox_inches='tight')
        plt.close()
    else:
        if show:
            plt.show()


def add_text_to_image(image, text):
    from PIL import ImageFont
    from PIL import ImageDraw
    
    # # resize image
    # image = image.resize((image.size[0] * 2, image.size[1] * 2))

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # font = ImageFont.load("arial.pil")
    # font = ImageFont.FreeTypeFont(size=20)
    # font = ImageFont.truetype("arial.ttf", 28, encoding="unic")

    # change fontsize
    
    # select color = black if image is mostly white
    if np.mean(image) > 200:
        draw.text((0, 0), text, (0,0,0), font=font)
    else:
        draw.text((0, 0), text, (255,255,255), font=font)
    
    # draw.text((0, 0), text, (255,255,255), font=font)
    return image


def show_keypoint_matches(
        img1, kp1, img2, kp2, matches,
        K=10, figsize=(10, 5), drawMatches_args=dict(matchesThickness=3, singlePointColor=(0, 0, 0)),
        choose_matches="random",
    ):
    """Displays matches found in the pair of images"""
    if choose_matches == "random":
        selected_matches = np.random.choice(matches, K)
    elif choose_matches == "all":
        K = len(matches)
        selected_matches = matches
    elif choose_matches == "topk":
        selected_matches = matches[:K]
    else:
        raise ValueError(f"Unknown value for choose_matches: {choose_matches}")

    # color each match with a different color
    cmap = matplotlib.cm.get_cmap('gist_rainbow', K)
    colors = [[int(x*255) for x in cmap(i)[:3]] for i in np.arange(0,K)]
    drawMatches_args.update({"matchColor": -1, "singlePointColor": (100, 100, 100)})
    
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, selected_matches, outImg=None, **drawMatches_args)
    show_single_image(
        img3,
        figsize=figsize,
        title=f"[{choose_matches.upper()}] Selected K = {K} matches between the pair of images.",
    )
    return img3


def draw_kps_on_image(image: np.ndarray, kps: np.ndarray, color=COLORS["red"], radius=5, thickness=-1, return_as="PIL"):
    """
    Draw keypoints on image.

    Args:
        image: Image to draw keypoints on.
        kps: Keypoints to draw. Note these should be in (x, y) format.
    """
    if isinstance(image, Image.Image):
        image = np.asarray(image)
    if isinstance(color, str):
        color = PIL.ImageColor.getrgb(color)
        colors = [color] * len(kps)
    elif isinstance(color, tuple):
        colors = [color] * len(kps)
    elif isinstance(color, list):
        colors = [PIL.ImageColor.getrgb(c) for c in color]
    assert len(colors) == len(kps), f"Number of colors ({len(colors)}) must be equal to number of keypoints ({len(kps)})"

    for kp, c in zip(kps, colors):
        image = cv2.circle(
            image.copy(), (int(kp[0]), int(kp[1])), radius=radius, color=c, thickness=thickness)
    
    if return_as == "PIL":
        return Image.fromarray(image)

    return image


def get_concat_h(im1, im2):
    """Concatenate two images horizontally"""
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def get_concat_v(im1, im2):
    """Concatenate two images vertically"""
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def show_images_with_keypoints(images: list, kps: list, radius=15, color=(0, 220, 220), figsize=(10, 8)):
    assert len(images) == len(kps)

    # generate
    images_with_kps = []
    for i in range(len(images)):
        img_with_kps = draw_kps_on_image(images[i], kps[i], radius=radius, color=color, return_as="PIL")
        images_with_kps.append(img_with_kps)
    
    # show
    show_grid_of_images(images_with_kps, n_cols=len(images), figsize=figsize)


def set_latex_fonts(usetex=True, fontsize=14, show_sample=False, **kwargs):
    try:
        plt.rcParams.update({
            "text.usetex": usetex,
            "font.family": "serif",
            # "font.serif": ["Computer Modern Romans"],
            "font.size": fontsize,
            **kwargs,
        })
        if show_sample:
            plt.figure()
            plt.title("Sample $y = x^2$")
            plt.plot(np.arange(0, 10), np.arange(0, 10)**2, "--o")
            plt.grid()
            plt.show()
    except:
        print("Failed to setup LaTeX fonts. Proceeding without.")
        pass



def plot_2d_points(
        list_of_points_2d,
        colors=None,
        sizes=None,
        markers=None,
        alpha=0.75,
        h=256,
        w=256,
        ax=None,
        save=True,
        savepath="test.png",
    ):

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.set_xlim([0, w])
    ax.set_ylim([0, h])
    
    if sizes is None:
        sizes = [0.1 for _ in range(len(list_of_points_2d))]
    if colors is None:
        colors = ["gray" for _ in range(len(list_of_points_2d))]
    if markers is None:
        markers = ["o" for _ in range(len(list_of_points_2d))]

    for points_2d, color, s, m in zip(list_of_points_2d, colors, sizes, markers):
        ax.scatter(points_2d[:, 0], points_2d[:, 1], s=s, alpha=alpha, color=color, marker=m)
    
    if save:
        plt.savefig(savepath, bbox_inches='tight')


def plot_2d_points_on_image(
        image,
        img_alpha=1.0,
        ax=None,
        list_of_points_2d=[],
        scatter_args=dict(),
    ):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.imshow(image, alpha=img_alpha)
    scatter_args["save"] = False
    plot_2d_points(list_of_points_2d, ax=ax, **scatter_args)
    
    # invert the axis
    ax.set_ylim(ax.get_ylim()[::-1])


def compare_landmarks(
        image, ground_truth_landmarks, v2d, predicted_landmarks,
        save=False, savepath="compare_landmarks.png", num_kps_to_show=-1,
        show_matches=True,
    ):

    # show GT landmarks on image
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    ax = axes[0]
    plot_2d_points_on_image(
        image,
        list_of_points_2d=[ground_truth_landmarks],
        scatter_args=dict(sizes=[15], colors=["limegreen"]),
        ax=ax,
    )
    ax.set_title("GT landmarks", fontsize=12)
    
    # since the projected points are inverted, using 180 degree rotation about z-axis
    ax = axes[1]
    plot_2d_points_on_image(
        image,
        list_of_points_2d=[v2d, predicted_landmarks],
        scatter_args=dict(sizes=[0.08, 15], markers=["o", "x"], colors=["royalblue", "red"]),
        ax=ax,
    )
    ax.set_title("Projection of predicted mesh", fontsize=12)
    
    # plot the ground truth and predicted landmarks on the same image
    ax = axes[2]
    plot_2d_points_on_image(
        image,
        list_of_points_2d=[
            ground_truth_landmarks[:num_kps_to_show],
            predicted_landmarks[:num_kps_to_show],
        ],
        scatter_args=dict(sizes=[15, 15], markers=["o", "x"], colors=["limegreen", "red"]),
        ax=ax,
        img_alpha=0.5,
    )
    ax.set_title("GT and predicted landmarks", fontsize=12)

    if show_matches:
        for i in range(num_kps_to_show):
            x_values = [ground_truth_landmarks[i, 0], predicted_landmarks[i, 0]]
            y_values = [ground_truth_landmarks[i, 1], predicted_landmarks[i, 1]]
            ax.plot(x_values, y_values, color="yellow", markersize=1, linewidth=2.)

    fig.tight_layout()
    if save:
        plt.savefig(savepath, bbox_inches="tight")
        


def plot_historgam_values(
        X, display_vals=False,
        bins=50, figsize=(8, 5),
        show_mean=True,
        xlabel=None, ylabel=None,
        ax=None, title=None, show=False,
        **kwargs,
    ):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.hist(X, bins=bins, **kwargs)
    if title is None:
        title = "Histogram of values"
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if display_vals:
        x, counts = np.unique(X, return_counts=True)
        # sort_indices = np.argsort(x)
        # x = x[sort_indices]
        # counts = counts[sort_indices]
        # for i in range(len(x)):
        #     ax.text(x[i], counts[i], counts[i], ha='center', va='bottom')
    
    ax.grid(alpha=0.3)
    
    if show_mean:
        mean = np.mean(X)
        mean_string = f"$\\mu$: {mean:.2f}"
        ax.set_title(title + f" ({mean_string}) ")
    else:
        ax.set_title(title)
    
    if not show:
        return ax
    else:
        plt.show()


"""Helper functions for all kinds of 2D/3D visualization"""
def bokeh_2d_scatter(x, y, desc, figsize=(700, 700), colors=None, use_nb=False, title="Bokeh scatter plot"):
    import matplotlib.colors as mcolors
    from bokeh.plotting import figure, output_file, show, ColumnDataSource
    from bokeh.models import HoverTool
    from bokeh.io import output_notebook

    if use_nb:
        output_notebook()

    # define colors to be assigned
    if colors is None:
        # applies the same color
        # create a color iterator: pick a random color and apply it to all points
        # colors = [np.random.choice(itertools.cycle(palette))] * len(x)
        colors = [np.random.choice(["red", "green", "blue", "yellow", "pink", "black", "gray"])] * len(x)

        # # applies different colors
        # colors = np.array([ [r, g, 150] for r, g in zip(50 + 2*x, 30 + 2*y) ], dtype="uint8")


    # define the df of data to plot
    source = ColumnDataSource(
            data=dict(
                x=x,
                y=y,
                desc=desc,
                color=colors,
            )
        )

    # define the attributes to show on hover
    hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("(x, y)", "($x, $y)"),
                ("Desc", "@desc"),
            ]
        )

    p = figure(
        plot_width=figsize[0], plot_height=figsize[1], tools=[hover], title=title,
    )
    p.circle('x', 'y', size=10, source=source, fill_color="color")
    show(p)




def bokeh_2d_scatter_new(
        df, x, y, hue, label, color_column=None, size_col=None,
        figsize=(650, 600), use_nb=False, title="Bokeh scatter plot",
        legend_loc="bottom_left", edge_color="black", audio_col=None,
    ):
    from bokeh.plotting import figure, output_file, show, ColumnDataSource
    from bokeh.models import HoverTool
    from bokeh.io import output_notebook

    if use_nb:
        output_notebook()

    assert {x, y, hue, label}.issubset(set(df.keys()))

    if isinstance(color_column, str) and color_column in df.keys():
        color_column_name = color_column
    else:
        import matplotlib.colors as mcolors
        colors = list(mcolors.BASE_COLORS.keys()) + list(mcolors.TABLEAU_COLORS.values())
        # colors = list(mcolors.BASE_COLORS.keys())
        colors = itertools.cycle(np.unique(colors))

        hue_to_color = dict()
        unique_hues = np.unique(df[hue].values)
        for _hue in unique_hues:
            hue_to_color[_hue] = next(colors)
        df["color"] = df[hue].apply(lambda k: hue_to_color[k])
        color_column_name = "color"
    
    if size_col is not None:
        assert isinstance(size_col, str) and size_col in df.keys()
    else:
        sizes = [10.] * len(df)
        df["size"] = sizes
        size_col = "size"

    source = ColumnDataSource(
        dict(
            x = df[x].values,
            y = df[y].values,
            hue = df[hue].values,
            label = df[label].values,
            color = df[color_column_name].values,
            edge_color = [edge_color] * len(df),
            sizes = df[size_col].values,
        )
    )

    # define the attributes to show on hover
    hover = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("(x, y)", "($x, $y)"),
                ("Desc", "@label"),
                ("Cluster", "@hue"),
            ]
        )

    p = figure(
        plot_width=figsize[0],
        plot_height=figsize[1],
        tools=["pan","wheel_zoom","box_zoom","save","reset","help"] + [hover],
        title=title,
    )
    p.circle(
        'x', 'y', size="sizes",
        source=source, fill_color="color",
        legend_group="hue", line_color="edge_color",
    )
    p.legend.location = legend_loc
    p.legend.click_policy="hide"


    show(p)

    
import torch
def get_sentence_embedding(model, tokenizer, sentence):
    encoded = tokenizer.encode_plus(sentence, return_tensors="pt")

    with torch.no_grad():
        output = model(**encoded)
    
    last_hidden_state = output.last_hidden_state
    assert last_hidden_state.shape[0] == 1
    assert last_hidden_state.shape[-1] == 768
    
    # only pick the [CLS] token embedding (sentence embedding)
    sentence_embedding = last_hidden_state[0, 0]
    
    return sentence_embedding


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def plot_histogram(df, col, ax=None, color="blue", title=None, xlabel=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.grid(alpha=0.3)
    xlabel = col if xlabel is None else xlabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    title = f"Historgam of {col}" if title is None else title
    ax.set_title(title)
    label = f"Mean: {np.round(df[col].mean(), 1)}"
    ax.hist(df[col].values, density=False, color=color, edgecolor=lighten_color(color, 0.1), label=label, **kwargs)
    if "bins" in kwargs:
        xticks = list(np.arange(kwargs["bins"])[::5])
        xticks += list(np.linspace(xticks[-1], int(df[col].max()), 5, dtype=int))
        # print(xticks)
        ax.set_xticks(xticks)
    ax.legend()
    plt.show()


def beautify_ax(ax, title=None, titlesize=20, sizealpha=0.7, xlabel=None, ylabel=None):
    labelsize = sizealpha * titlesize
    ax.grid(alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    ax.set_title(title, fontsize=titlesize)




def get_text_features(text: list, model, device, batch_size=16):
    import clip
    text_batches = [text[i:i+batch_size] for i in range(0, len(text), batch_size)]
    text_features = []
    model = model.to(device)
    model = model.eval()
    for batch in tqdm(text_batches, desc="Getting text features", bar_format="{l_bar}{bar:20}{r_bar}"):
        batch = clip.tokenize(batch).to(device)
        with torch.no_grad():
            batch_features = model.encode_text(batch)
        text_features.append(batch_features.cpu().numpy())
    text_features = np.concatenate(text_features, axis=0)
    return text_features


from sklearn.manifold import TSNE
def reduce_dim(X, method="tsne", perplexity=30, n_iter=1000):
    if method == "tsne":
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            # n_iter=n_iter,
            init='pca',
            random_state=42,
            # learning_rate="auto",
        )
        Z = tsne.fit_transform(X)
    elif method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        Z = pca.fit_transform(X)
    elif method == "umap":
        import umap
        reducer = umap.UMAP(random_state=42)
        Z = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unknown method {method}")
    return Z


try:
    from IPython.display import Video
except ImportError:
    Video = None
def show_video(video_path):
    """Show a video in a Jupyter notebook"""
    assert exists(video_path), f"Video path {video_path} does not exist"
    
    # display the video in a Jupyter notebook
    return Video(video_path, embed=True, width=480)
    # Video(video_path, embed=True, width=600, height=400)
    # html_attributes="controls autoplay loop muted"




def show_single_audio(filepath=None, data=None, rate=None, start=None, end=None, label="Sample audio"):
    import librosa
    
    if filepath is None:
        assert data is not None and rate is not None, "Either filepath or data and rate must be provided"
        args = dict(data=data, rate=rate)
    else:
        assert data is None and rate is None, "Either filepath or data and rate must be provided"
        data, rate = librosa.load(filepath)
        # args = dict(filename=filepath)
        args = dict(data=data, rate=rate)
    
    if start is not None and end is not None:
        start = max(int(start * rate), 0)
        end = min(int(end * rate), len(data))
    else:
        start = 0
        end = len(data)
    data = data[start:end]
    args["data"] = data

    if label is None:
        label = "Sample audio"

    # Use HTML widget with width constraints and text wrapping
    # Set label width to match audio widget width (with some padding)
    label_width = 400  # Approximate width for audio widgets
    label_html = f'<div style="width: {label_width}px; word-wrap: break-word; overflow-wrap: break-word;">{label}</div>'
    label_widget = HTML(value=label_html)
    
    out = widgets.Output()
    with out:
        display(Audio(**args))
    vbox = VBox([label_widget, out])
    return vbox


def show_single_audio_with_spectrogram(filepath=None, data=None, rate=None, label="Sample audio", figsize=(6, 2)):
    import librosa
    if filepath is None:
        assert data is not None and rate is not None, "Either filepath or data and rate must be provided"
    else:
        data, rate = librosa.load(filepath)
    
    # Show audio
    vbox = show_single_audio(data=data, rate=rate, label=label)
    # get width of audio widget
    width = vbox.children[1].layout.width

    # Show spectrogram
    spec_out = widgets.Output()
    D = librosa.stft(data)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    with spec_out:
        fig, ax = plt.subplots(figsize=figsize)
        img = librosa.display.specshow(
            S_db,
            ax=ax,
            x_axis='time',
            # y_axis='linear',
        )
    # img = widgets.Image.from_file(fig)
    # import ipdb; ipdb.set_trace()
    # img = widgets.Image(img)
    # add image to vbox
    vbox.children += (spec_out,)
    return vbox

def show_spectrogram(audio_path=None, data=None, rate=None, figsize=(6, 2), ax=None, show=True):
    import librosa
    if data is None and rate is None:
        # Show spectrogram
        data, rate = librosa.load(audio_path)
    else:
        assert audio_path is None, "Either audio_path or data and rate must be provided"

    hop_length = 512
    D = librosa.stft(data, n_fft=2048, hop_length=hop_length, win_length=2048)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Create spectrogram plot widget
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(S_db, origin='lower', aspect='auto', cmap='inferno')

    # Replace xtixks with time
    xticks = ax.get_xticks()
    time_in_seconds = librosa.frames_to_time(xticks, sr=rate, hop_length=hop_length)
    ax.set_xticklabels(np.round(time_in_seconds, 1))
    ax.set_xlabel('Time')
    ax.set_yticks([])
    if ax is None:
        plt.close(fig)

    # Create widget output
    spec_out = widgets.Output()
    with spec_out:
        display(fig)
    return spec_out


def show_single_video_and_spectrogram(
        video_path, audio_path,
        label="Sample video", figsize=(6, 2),
        width=480,
        show_spec_stats=False,
    ):
    import librosa
    # Show video
    vbox = show_single_video(video_path, label=label, width=width)
    # get width of video widget
    width = vbox.children[1].layout.width

    # Show spectrogram
    data, rate = librosa.load(audio_path)
    hop_length = 512
    D = librosa.stft(data, n_fft=2048, hop_length=hop_length, win_length=2048)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Create spectrogram plot widget
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(S_db, origin='lower', aspect='auto', cmap='inferno')

    # Replace xtixks with time
    xticks = ax.get_xticks()
    time_in_seconds = librosa.frames_to_time(xticks, sr=rate, hop_length=hop_length)
    ax.set_xticklabels(np.round(time_in_seconds, 1))
    ax.set_xlabel('Time')
    ax.set_yticks([])
    plt.close(fig)

    # Create widget output
    spec_out = widgets.Output()
    with spec_out:
        display(fig)
    vbox.children += (spec_out,)

    if show_spec_stats:
        # Compute mean of spectrogram over frequency axis
        eps = 1e-5
        S_db_normalized = (S_db - S_db.mean(axis=1)[:, None]) / (S_db.std(axis=1)[:, None] + eps)
        S_db_over_time = S_db_normalized.sum(axis=0)
        # Plot S_db_over_time
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        # ax.set_title("Spectrogram over time")
        ax.grid(alpha=0.5)
        x = np.arange(len(S_db_over_time))
        x = librosa.frames_to_time(x, sr=rate, hop_length=hop_length)
        x = np.round(x, 1)
        ax.plot(x, S_db_over_time)
        ax.set_xlabel('Time')
        ax.set_yticks([])
        plt.close(fig)
        plot_out = widgets.Output()
        with plot_out:
            display(fig)
        vbox.children += (plot_out,)

    return vbox


def show_single_spectrogram(
        filepath=None,
        data=None,
        rate=None,
        start=None,
        end=None,
        ax=None,
        label="Sample spectrogram",
        figsize=(6, 2),
        xlabel="Time",
    ):
    import librosa
    
    if filepath is None:
        assert data is not None and rate is not None, "Either filepath or data and rate must be provided"
    else:
        rate = 22050
        offset = start or 0
        clip_duration = end - start if end is not None else None
        data, rate = librosa.load(filepath, sr=rate, offset=offset, duration=clip_duration)
    
    # start = 0 if start is None else int(rate * start)
    # end = len(data) if end is None else int(rate * end)
    # data = data[start:end]
    
    # Show spectrogram
    spec_out = widgets.Output()
    D = librosa.stft(data)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    with spec_out:
        img = librosa.display.specshow(
            S_db,
            ax=ax,
            x_axis='time',
            sr=rate,
            # y_axis='linear',
        )
    ax.set_xlabel(xlabel)
    ax.margins(x=0)
    plt.subplots_adjust(wspace=0, hspace=0)

    # img = widgets.Image.from_file(fig)
    # import ipdb; ipdb.set_trace()
    # img = widgets.Image(img)
    # add image to vbox
    vbox = VBox([spec_out])
    return vbox
    # return spec_out


# from decord import VideoReader
def show_single_video(filepath, label="Sample video", width=480, fix_resolution=True):
    
    if label is None:
        label = "Sample video"
    
    height = None
    if fix_resolution:
        aspect_ratio = 16. / 9.
        height = int(width * (1/ aspect_ratio))

    # Use HTML widget with width constraints and text wrapping
    # Set label width to match video width (with some padding)
    label_width = width - 20  # Leave some padding
    label_html = f'<div style="width: {label_width}px; word-wrap: break-word; overflow-wrap: break-word; line-height: 1.2; margin: 0; padding: 0;">{label}</div>'
    label_widget = HTML(value=label_html)
    
    out = widgets.Output()
    with out:
        display(Video(filepath, embed=True, width=width, height=height))
    # vbox = VBox([label_widget, out])
    vbox = VBox([out, label_widget])
    return vbox



def color_text(text: str, color: str):
    from ipywidgets import widgets
    htmlWidget = widgets.HTML(value = f"<font color='{color}'>{text}")
    return htmlWidget


def show_colored_text(text: list, colors: list):
    """
    Display Label() widgets with different colors.
    
    Args:
        text: list of strings
        colors: list of colors
    
    The output must be <text[0] with colors[0] <text[1] with colors[1] ...

    Returns:
        VBox() widget
    """
    from ipywidgets import Label
    from IPython.display import display
    from ipywidgets import VBox
    from ipywidgets import Layout

    # Create a list of Label widgets with different colors
    # labels = [Label(value=f"{t}", layout=Layout(color=c)) for t, c in zip(text, colors)]
    labels = [color_text(t, c) for t, c in zip(text, colors)]
    
    # Print the output in a single line
    hbox = HBox(labels)
    display(hbox)




def show_single_image_sequence(
        filepath, n_frames=4, label="Sample image sequence",
        width=480, fix_resolution=True, max_width=1000, reverse=False,
    ):
    # if label is None:
    #     label = "Sample image sequence"
    
    height = None
    if fix_resolution:
        aspect_ratio = 16. / 9.
        height = int(width * (1/ aspect_ratio))
    
    if label is not None:
        # Use HTML widget with width constraints and text wrapping
        # Set label width to match the canvas width (with some padding)
        label_width = min(max_width, width) - 20  # Leave some padding
        label_html = f'<div style="width: {label_width}px; word-wrap: break-word; overflow-wrap: break-word;">{label}</div>'
        label_widget = HTML(value=label)
        # label_widget = HTML(value=label_html)
    out = widgets.Output()
    
    # Load frames to show
    from decord import VideoReader
    vr = VideoReader(filepath, num_threads=1)
    n_frames = min(n_frames, len(vr))
    # frames = [vr[i] for i in range(0, len(vr), len(vr) // n_frames)]
    indices = np.linspace(0, len(vr)-1, n_frames)
    if reverse:
        indices = indices[::-1]
    frames = vr.get_batch(indices).asnumpy()
    frames = [Image.fromarray(f) for f in frames]
    canvas = concat_images(frames)

    # Resize canvas to width max_width
    if canvas.size[0] > max_width:
        canvas = canvas.resize((max_width, int(max_width * canvas.size[1] / canvas.size[0])))

    with out:
        display(canvas)
    
    if label is not None:
        vbox = VBox([label_widget, out])
    else:
        vbox = out
    return vbox


def show_grid_of_audio(files, starts=None, ends=None, labels=None, ncols=None, show_spec=False):
    
    for f in files:
        assert os.path.exists(f), f"File {f} does not exist."

    if labels is None:
        labels = [None] * len(files)
    
    if starts is None:
        starts = [None] * len(files)
    
    if ends is None:
        ends = [None] * len(files)

    assert len(files) == len(labels)
    
    if ncols is None:
        ncols = 3
    nfiles = len(files)
    nrows = nfiles // ncols + (nfiles % ncols != 0)
    # print(nrows, ncols)
    
    for i in range(nrows):
        row_hbox = []
        for j in range(ncols):
            idx = i * ncols + j
            # print(i, j, idx)
            
            if idx < len(files):
                file, label = files[idx], labels[idx]
                start, end = starts[idx], ends[idx]
                vbox = show_single_audio(
                    filepath=file, label=label, start=start, end=end
                )
                if show_spec:
                    spec_box = show_spectrogram(file, figsize=(3.6, 1))
                    # Add spectrogram to vbox
                    vbox.children += (spec_box,)

                # if not show_spec:
                #     vbox = show_single_audio(
                #         filepath=file, label=label, start=start, end=end
                #     )
                # else:
                #     vbox = show_single_audio_with_spectrogram(
                #         filepath=file, label=label
                #     )
                row_hbox.append(vbox)
        row_hbox = HBox(row_hbox)
        display(row_hbox)


def show_grid_of_videos(
        files,
        cut=False,
        starts=None,
        ends=None,
        labels=None,
        ncols=None,
        width_overflow=False,
        show_spec=False,
        width_of_screen=1000,
    ):
    from moviepy.editor import VideoFileClip
    
    for f in files:
        assert os.path.exists(f), f"File {f} does not exist."

    if labels is None:
        labels = [None] * len(files)
    if starts is not None and ends is not None:
        cut = True
    if starts is None:
        starts = [None] * len(files)
    if ends is None:
        ends = [None] * len(files)

    assert len(files) == len(labels) == len(starts) == len(ends)
    
    # cut the videos to the specified duration
    if cut:
        cut_files = []
        for i, f in enumerate(files):
            start, end = starts[i], ends[i]
            
            tmp_f = os.path.join(os.path.expanduser("~"), f"tmp/clip_{i}.mp4")
            cut_files.append(tmp_f)
        
            video = VideoFileClip(f)
            start = 0 if start is None else start
            end = video.duration-1 if end is None else end
            # print(start, end)
            video.subclip(start, end).write_videofile(tmp_f, logger=None, verbose=False)
        files = cut_files

    if ncols is None:
        ncols = 3
        width_of_screen = 1000

    # get width of the whole display screen
    if not width_overflow:
        width_of_single_video = width_of_screen // ncols
    else:
        width_of_single_video = 280

    nfiles = len(files)
    nrows = nfiles // ncols + (nfiles % ncols != 0)
    # print(nrows, ncols)
    
    for i in range(nrows):
        row_hbox = []
        for j in range(ncols):
            idx = i * ncols + j
            # print(i, j, idx)
            
            if idx < len(files):
                file, label = files[idx], labels[idx]
                if not show_spec:
                    vbox = show_single_video(file, label, width_of_single_video)
                else:
                    vbox = show_single_video_and_spectrogram(file, file, width=width_of_single_video, label=label)
                row_hbox.append(vbox)
        row_hbox = HBox(row_hbox)
        display(row_hbox)


def convert_video_to_gif(file, tmp_f, reverse=False):
    from moviepy.editor import VideoFileClip, vfx
    video = VideoFileClip(file)
    if reverse:
        video = video.fx(vfx.time_mirror)  # reverse in time
    video.write_gif(tmp_f, logger=None, verbose=False)
    return tmp_f



def show_grid_of_gifs(files, labels=None, ncols=None, width_of_screen=1000):
    for f in files:
        assert os.path.exists(f), f"File {f} does not exist."

    if labels is None:
        labels = [None] * len(files)

    if ncols is None:
        ncols = 3
    nfiles = len(files)
    nrows = nfiles // ncols + (nfiles % ncols != 0)
    # print(nrows, ncols)
    
    width_of_single_video = width_of_screen // ncols
    
    for i in range(nrows):
        row_hbox = []
        for j in range(ncols):
            idx = i * ncols + j
            # print(i, j, idx)
            if idx < len(files):
                file, label = files[idx], labels[idx]
                vbox = show_single_gif(file, label, width_of_single_video)
                row_hbox.append(vbox)
        row_hbox = HBox(row_hbox)
        display(row_hbox)


def show_single_gif(file, label, width_of_single_video):
    from ipywidgets import Image as WidgetImage, VBox, Label
    
    # Create the image widget
    image_widget = WidgetImage(value=open(file, 'rb').read(), width=width_of_single_video)
    
    # Create the label widget if label is provided
    if label is not None:
        label_widget = Label(value=str(label))
        return VBox([image_widget, label_widget])
    else:
        return image_widget


def show_grid_of_videos_as_gifs(files, labels=None, reverse=None,ncols=None, width_of_screen=1000):
    reverse = [False] * len(files) if reverse is None else reverse
    
    # First, convert all the files to gifs in the tmp directory 
    tmp_dir = os.path.join(os.path.expanduser("~"), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_files = []
    for i, f in enumerate(files):
        tmp_f = os.path.join(tmp_dir, f"clip_{i}.gif")
        convert_video_to_gif(f, tmp_f, reverse=reverse[i])
        tmp_files.append(tmp_f)

    # Then, show the gifs in a grid
    show_grid_of_gifs(tmp_files, labels, ncols, width_of_screen)
    
    return tmp_files


def show_grid_of_image_sequences(
        files,
        cut=False,
        starts=None,
        ends=None,
        labels=None,
        ncols=None,
        width_overflow=False,
        show_spec=False,
        width_of_screen=1200,
        n_frames=4,
    ):
    from moviepy.editor import VideoFileClip
    
    for f in files:
        assert os.path.exists(f), f"File {f} does not exist."

    if labels is None:
        labels = [None] * len(files)
    if starts is not None and ends is not None:
        cut = True
    if starts is None:
        starts = [None] * len(files)
    if ends is None:
        ends = [None] * len(files)

    assert len(files) == len(labels) == len(starts) == len(ends)
    
    # cut the videos to the specified duration
    if cut:
        cut_files = []
        for i, f in enumerate(files):
            start, end = starts[i], ends[i]
            
            tmp_f = os.path.join(os.path.expanduser("~"), f"tmp/clip_{i}.mp4")
            cut_files.append(tmp_f)
        
            video = VideoFileClip(f)
            start = 0 if start is None else start
            end = video.duration-1 if end is None else end
            # print(start, end)
            video.subclip(start, end).write_videofile(tmp_f, logger=None, verbose=False)
        files = cut_files

    if ncols is None:
        ncols = 3
        width_of_screen = 1000

    # get width of the whole display screen
    if not width_overflow:
        width_of_single_video = width_of_screen // ncols
    else:
        width_of_single_video = 280

    nfiles = len(files)
    nrows = nfiles // ncols + (nfiles % ncols != 0)
    # print(nrows, ncols)
    
    for i in range(nrows):
        row_hbox = []
        for j in range(ncols):
            idx = i * ncols + j
            # print(i, j, idx)
            
            if idx < len(files):
                file, label = files[idx], labels[idx]
                if not show_spec:
                    vbox = show_single_image_sequence(
                        file, n_frames, label, max_width=(width_of_single_video * n_frames),
                    )
                else:
                    raise NotImplementedError
                row_hbox.append(vbox)
        row_hbox = HBox(row_hbox)
        display(row_hbox)


def preview_video(fp, label="Sample video frames", mode="uniform", frames_to_show=6):
    from decord import VideoReader
    
    assert exists(fp), f"Video does not exist at {fp}"
    vr = VideoReader(fp)

    nfs = len(vr)
    fps = vr.get_avg_fps()
    dur = nfs / fps
    
    if mode == "all":
        frame_indices = np.arange(nfs)
    elif mode == "uniform":
        frame_indices = np.linspace(0, nfs - 1, frames_to_show, dtype=int)
    elif mode == "random":
        frame_indices = np.random.randint(0, nfs - 1, replace=False)
        frame_indices = sorted(frame_indices)
    else:
        raise ValueError(f"Unknown frame viewing mode {mode}.")
    
    # Show grid of image
    images = vr.get_batch(frame_indices).asnumpy()
    show_grid_of_images(images, n_cols=len(frame_indices), title=label, figsize=(12, 2.3), titlesize=10)


def preview_multiple_videos(fps, labels, mode="uniform", frames_to_show=6):
    for fp in fps:
        assert exists(fp), f"Video does not exist at {fp}"
    
    for fp, label in zip(fps, labels):
        preview_video(fp, label, mode=mode, frames_to_show=frames_to_show)



def show_small_clips_in_a_video(
        video_path,
        clip_segments: list,
        width=360,
        labels=None,
        show_spec=False,
        resize=False,
    ):
    from moviepy.editor import VideoFileClip
    from ipywidgets import Layout

    video = VideoFileClip(video_path)
    
    if resize:
        # Resize the video
        print("Resizing the video to width", width)
        video = video.resize(width=width)
    
    if labels is None:
        labels = [
            f"Clip {i+1} [{clip_segments[i][0]} : {clip_segments[i][1]}]" for i in range(len(clip_segments))
        ]
    else:
        assert len(labels) == len(clip_segments)
    
    tmp_dir = os.path.join(os.path.expanduser("~"), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_clippaths = [f"{tmp_dir}/clip_{i}.mp4" for i in range(len(clip_segments))]
    
    iterator = tqdm_iterator(zip(clip_segments, tmp_clippaths), total=len(clip_segments), desc="Preparing clips")
    clips = [
        video.subclip(x, y).write_videofile(f, logger=None, verbose=False) \
            for (x, y), f in iterator
    ]
    # show_grid_of_videos(tmp_clippaths, labels, ncols=len(clips), width_overflow=True)
    hbox = []
    for i in range(len(clips)):
        # vbox = show_single_video(tmp_clippaths[i], labels[i], width=280)
        
        vbox = widgets.Output()
        with vbox:
            if show_spec:
                display(
                    show_single_video_and_spectrogram(
                        tmp_clippaths[i], tmp_clippaths[i],
                        width=width, figsize=(4.4, 1.5), 
                    )
                )
            else:
                display(Video(tmp_clippaths[i], embed=True, width=width))
            # reduce vspace between video and label
            display(Label(labels[i], layout=Layout(margin="-8px 0px 0px 0px")))
            # if show_spec:
            #     display(show_single_spectrogram(tmp_clippaths[i], figsize=(4.5, 1.5)))
        hbox.append(vbox)
    hbox = HBox(hbox)
    display(hbox)


def show_single_video_and_audio(
        video_path, audio_path, label="Sample video and audio",
        start=None, end=None, width=360, sr=44100, show=True,
    ):
    from moviepy.editor import VideoFileClip
    import librosa

    # Load video
    video = VideoFileClip(video_path)
    video_args = {"embed": True, "width": width}
    filepath = video_path

    # Load audio
    audio_waveform, sr = librosa.load(audio_path, sr=sr)
    audio_args = {"data": audio_waveform, "rate": sr}

    if start is not None and end is not None:
        
        # Cut video from start to end
        tmp_dir = os.path.join(os.path.expanduser("~"), "tmp")
        clip_path = os.path.join(tmp_dir, "clip_sample.mp4")
        video.subclip(start, end).write_videofile(clip_path, logger=None, verbose=False)
        filepath = clip_path
        
        # Cut audio from start to end
        audio_waveform = audio_waveform[int(start * sr): int(end * sr)]
        audio_args["data"] = audio_waveform

    out = widgets.Output()
    with out:
        label_text = f"{label} [{start} : {end}]"
        # Use HTML widget with width constraints and text wrapping
        label_width = width - 20  # Leave some padding
        label_html = f'<div style="width: {label_width}px; word-wrap: break-word; overflow-wrap: break-word; line-height: 1.2; margin: 0; padding: 0;">{label_text}</div>'
        label_widget = HTML(value=label_html)
        display(label_widget)
        display(Video(filepath, **video_args))
        display(Audio(**audio_args))
    
    if show:
        display(out)
    else:
        return out


def plot_waveform(waveform, sample_rate, figsize=(10, 2), ax=None, skip=100, show=True, title=None):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    time_axis = torch.arange(0, len(waveform)) / sample_rate
    waveform = waveform[::skip]
    time_axis = time_axis[::skip]

    if len(waveform.shape) == 1:
        num_channels = 1
        num_frames = waveform.shape[0]
        waveform = waveform.reshape(1, num_frames)
    elif len(waveform.shape) == 2:
        num_channels, num_frames = waveform.shape
    else:
        raise ValueError(f"Waveform has invalid shape {waveform.shape}")
    
    if ax is None:
        figure, axes = plt.subplots(num_channels, 1, figsize=figsize)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")
        figure.suptitle(title)
    else:
        assert num_channels == 1
        ax.plot(time_axis, waveform[0], linewidth=1)
        ax.grid(True)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.05, 0.05)
    
    if show:
        plt.show(block=False)


def show_waveform_as_image(waveform, sr=16000):
    """Plots a waveform as plt fig and converts into PIL.Image"""
    fig, ax = plt.subplots(figsize=(10, 2))
    plot_waveform(waveform, sr, ax=ax, show=False)
    fig.canvas.draw()
    img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)
    return img


def plot_raw_audio_signal_with_markings(signal: np.ndarray, markings: list,
        title: str = 'Raw audio signal with markings',
        figsize: tuple = (23, 4),
    ):

    plt.figure(figsize=figsize)
    plt.grid()

    plt.plot(signal)
    for value in markings:
        plt.axvline(x=value, c='red')
    plt.xlabel('Time')
    plt.title(title)

    plt.show()
    plt.close()


def get_concat_h(im1, im2):
    """Concatenate two images horizontally"""
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def concat_images(images):
    im1 = images[0]
    dst = Image.new('RGB', (sum([im.width for im in images]), im1.height))
    start_width = 0
    for i, im in enumerate(images):
        dst.paste(im, (start_width, 0))
        start_width += im.width
    return dst


def concat_images_with_border(images, border_width=5, border_color="white"):
    im1 = images[0]
    dst = Image.new('RGB', (sum([im.width for im in images]) + (len(images) - 1) * border_width, im1.height), border_color)
    start_width = 0
    uniform_height = im1.height
    for i, im in enumerate(images):
        # if im.height != uniform_height:
        #     im = resize_height(im.copy(), uniform_height)
        dst.paste(im, (start_width, 0))
        start_width += im.width + border_width
    return dst


def concat_images_vertically(images):
    im1 = images[0]
    dst = Image.new('RGB', (im1.width, sum([im.height for im in images])))
    start_height = 0
    for i, im in enumerate(images):
        dst.paste(im, (0, start_height))
        start_height += im.height
    return dst


def concat_images_vertically_with_border(images, border_width=5, border_color="white"):
    im1 = images[0]
    dst = Image.new('RGB', (im1.width, sum([im.height for im in images]) + (len(images) - 1) * border_width), border_color)
    start_height = 0
    for i, im in enumerate(images):
        dst.paste(im, (0, start_height))
        start_height += im.height + border_width
    return dst


def get_concat_v(im1, im2):
    """Concatenate two images vertically"""
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def set_latex_fonts(usetex=True, fontsize=14, show_sample=False, **kwargs):
    try:
        plt.rcParams.update({
            "text.usetex": usetex,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": fontsize,
            **kwargs,
        })
        if show_sample:
            plt.figure()
            plt.title("Sample $y = x^2$")
            plt.plot(np.arange(0, 10), np.arange(0, 10)**2, "--o")
            plt.grid()
            plt.show()
    except:
        print("Failed to setup LaTeX fonts. Proceeding without.")
        pass


def get_colors(num_colors, palette="jet"):
    cmap = plt.get_cmap(palette)
    colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    return colors


def add_box_on_image(image, bbox, color="red", thickness=3, resized=False, fillcolor=None, fillalpha=0.2):
    """
    Adds bounding box on image.
    
    Args:
        image (PIL.Image): image
        bbox (list): [xmin, ymin, xmax, ymax]
        color: -
        thickness: -
    """
    image = image.copy().convert("RGB")
    # color = get_predominant_color(color)
    color = PIL.ImageColor.getrgb(color)
    
    # Apply alpha to fillcolor
    if fillcolor is not None:
        if isinstance(fillcolor, str):
            fillcolor = PIL.ImageColor.getrgb(fillcolor)
            fillcolor= fillcolor + (int(fillalpha * 255),)
        elif isinstance(fillcolor, tuple):
            if len(fillcolor) == 3:
                fillcolor= fillcolor + (int(fillalpha * 255),)
            else:
                pass

    # Create an instance of the ImageDraw class
    draw = ImageDraw.Draw(image, "RGBA")

    # Draw the bounding box on the image
    draw.rectangle(bbox, outline=color, width=thickness, fill=fillcolor)

    # Resize
    new_width, new_height = (320, 240)
    if resized:
        image = image.resize((new_width, new_height))

    return image


def add_multiple_boxes_on_image(image, bboxes, colors=None, thickness=3, resized=False, fillcolor=None, fillalpha=0.2):
    image = image.copy().convert("RGB")
    if colors is None:
        colors = ["red"] * len(bboxes)
    for bbox, color in zip(bboxes, colors):
        image = add_box_on_image(image, bbox, color, thickness, resized, fillcolor, fillalpha)
    return image


def colorize_mask(mask, color="red"):
    # mask = mask.convert("RGBA")
    color = PIL.ImageColor.getrgb(color)
    mask = ImageOps.colorize(mask, (0, 0, 0, 0), color)
    return mask


def convert_mask_to_image(mask, threshold=0.5):
    """Converts a numpy array or torch tensor between [0, 1] to a PIL image."""
    binary_array = (mask > threshold).astype(np.uint8) * 255
    binary_array = np.clip(binary_array, 0, 255)
    binary_image = Image.fromarray(binary_array, mode="L")
    return binary_image


def add_mask_on_image(image: Image, mask: Image, color="green", alpha=0.5):

    if isinstance(mask, np.ndarray):
        mask = convert_mask_to_image(mask)

    image = image.copy()
    mask = mask.copy()

    # get color if it is a string
    if isinstance(color, str):
        color = PIL.ImageColor.getrgb(color)
    # color = get_predominant_color(color)
    mask = ImageOps.colorize(mask, (0, 0, 0, 0), color)

    mask = mask.convert("RGB")
    assert (mask.size == image.size)
    assert (mask.mode == image.mode)

    # Blend the original image and the segmentation mask with a 50% weight
    blended_image = Image.blend(image, mask, alpha)
    return blended_image


def blend_images(img1, img2, alpha=0.5):
    # Convert images to RGBA
    img1 = img1.convert("RGBA")
    img2 = img2.convert("RGBA")
    alpha_blended = Image.blend(img1, img2, alpha=alpha)
    # Convert back to RGB
    alpha_blended = alpha_blended.convert("RGB")
    return alpha_blended


def visualize_youtube_clip(
        youtube_id, st, et, label="",
        show_spec=False,
        video_width=360, video_height=240,
    ):
    import librosa
    
    url = f"https://www.youtube.com/embed/{youtube_id}?start={int(st)}&end={int(et)}"
    video_html_code = f"""
    <iframe height="{video_height}" width="{video_width}" src="{url}" frameborder="0" allowfullscreen></iframe>
    """
    label_html_code = f"""<b>Caption</b>: {label} <br> <b>Time</b>: {st} to {et}"""
    
    # Show label and video below it
    label = widgets.HTML(label_html_code)
    video = widgets.HTML(video_html_code)
    
    if show_spec:
        import pytube
        import base64
        from io import BytesIO
        from moviepy.video.io.VideoFileClip import VideoFileClip
        from moviepy.audio.io.AudioFileClip import AudioFileClip

        # Load audio directly from youtube
        video_url = f"https://www.youtube.com/watch?v={youtube_id}"
        yt = pytube.YouTube(video_url)
        # Get the audio stream
        audio_stream = yt.streams.filter(only_audio=True).first()

        # Download audio stream
        # audio_file = os.path.join("/tmp", "sample_audio.mp3")
        audio_stream.download(output_path='/tmp', filename='sample.mp4')
        
        audio_clip = AudioFileClip("/tmp/sample.mp4")
        audio_subclip = audio_clip.subclip(st, et)
        sr = audio_subclip.fps
        y = audio_subclip.to_soundarray().mean(axis=1)
        audio_subclip.close()
        audio_clip.close()
        
        # Compute spectrogram in librosa
        S_db = librosa.power_to_db(librosa.feature.melspectrogram(y, sr=sr), ref=np.max)
        # Compute width in cms from video_width
        width = video_width / plt.rcParams["figure.dpi"] + 0.63
        height = video_height / plt.rcParams["figure.dpi"]
        out = widgets.Output()
        with out:
            fig, ax = plt.subplots(figsize=(width, height))
            librosa.display.specshow(S_db, sr=sr, x_axis='time', ax=ax)
            ax.set_ylabel("Frequency (Hz)")
    else:
        out = widgets.Output()
    
    vbox = widgets.VBox([label, video, out])

    return vbox
 

def visualize_pair_of_youtube_clips(clip_a, clip_b):
    yt_id_a = clip_a["youtube_id"]
    label_a = clip_a["sentence"]
    st_a, et_a = clip_a["time"]
    
    yt_id_b = clip_b["youtube_id"]
    label_b = clip_b["sentence"]
    st_b, et_b = clip_b["time"]
    
    # Show the clips side by side
    clip_a = visualize_youtube_clip(yt_id_a, st_a, et_a, label_a, show_spec=True)
    # clip_a = widgets.Output()
    # with clip_a:
    #     visualize_youtube_clip(yt_id_a, st_a, et_a, label_a, show_spec=True)
    
    clip_b = visualize_youtube_clip(yt_id_b, st_b, et_b, label_b, show_spec=True)
    # clip_b = widgets.Output()
    # with clip_b:
    #     visualize_youtube_clip(yt_id_b, st_b, et_b, label_b, show_spec=True)

    hbox = HBox([
        clip_a, clip_b
    ])
    display(hbox)
    

def plot_1d(x: np.ndarray, figsize=(6, 2), title=None, xlabel=None, ylabel=None, show=True, **kwargs):
    assert (x.ndim == 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(np.arange(len(x)), x, **kwargs)
    if show:
        plt.show()
    else:
        plt.close()
    return fig



def make_grid(cols,rows):
    import streamlit as st
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid


def display_clip(video_path, stime, etime, label=None):
    """Displays clip at index i."""
    assert exists(video_path), f"Video does not exist at {video_path}"
    display(
        show_small_clips_in_a_video(
            video_path, [(stime, etime)], labels=[label],
        ),
    )


def countplot(df, column, title=None, rotation=90, ylabel="Count", figsize=(8, 5), ax=None, show=True, show_counts=False):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.grid(alpha=0.4)
    ax.set_xlabel(column)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    data = dict(df[column].value_counts())
    # Extract keys and values from the dictionary
    categories = list(data.keys())
    counts = list(data.values())

    # Create a countplot
    ax.bar(categories, counts)
    ax.set_xticklabels(categories, rotation=rotation)
    
    # Show count values on top of bars
    if show_counts:
        max_v = max(counts)
        for i, v in enumerate(counts):
            delta = 0.01 * max_v
            ax.text(i, v + delta, str(v), ha="center")
    
    if show:
        plt.show()


def get_linspace_colors(cmap_name='viridis', num_colors = 10):
    import matplotlib.colors as mcolors

    # Get the colormap object
    cmap = plt.cm.get_cmap(cmap_name)

    # Get the evenly spaced indices
    if num_colors == 1:
        indices = [0.5]
    elif num_colors == 2:
        indices = [0.1, 0.9]
    else:
        gap = 1 / (num_colors)
        indices = np.arange(0, 1, gap)

    # Get the corresponding colors from the colormap
    colors = [mcolors.to_hex(cmap(idx)) for idx in indices]
    
    return colors


def hex_to_rgb(colors):
    from PIL import ImageColor
    return [ImageColor.getcolor(c, "RGB") for c in colors]


def plot_audio_feature(times, feature, feature_label="Feature", xlabel="Time", figsize=(20, 2)):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.grid(alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(feature_label)
    ax.set_yticks([])
    
    ax.plot(times, feature, '--', linewidth=0.5)
    plt.show()



def compute_rms(y, frame_length=512):
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=frame_length)[0]
    times = librosa.samples_to_time(frame_length * np.arange(len(rms)))
    return times, rms


def plot_audio_features(path, label, show=True, show_video=True, features=["rms"], frame_length=512, figsize=(5, 2), return_features=False):
    import librosa
    # Load audio
    y, sr = librosa.load(path)
    
    # Show video
    if show_video:
        if show:
            display(
                show_single_video_and_spectrogram(
                    path, path, label=label, figsize=figsize,
                    width=410,
                )
            )
    else:
        if show:
            # Show audio and spectrogram
            display(
                show_single_audio_with_spectrogram(path, label=label, figsize=figsize)
            )

    feature_data = dict() 
    for f in features:
        fn = eval(f"compute_{f}")
        args = dict(y=y, frame_length=frame_length)
        xvals, yvals = fn(**args)
        feature_data[f] = (xvals, yvals)
        
        if show:
            display(
                plot_audio_feature(
                    xvals, yvals, feature_label=f.upper(), figsize=(figsize[0] - 0.25, figsize[1]),
                )
            )
    
    if return_features:
        return feature_data


def rescale_frame(frame, scale=1.):
    """Rescales a frame by a factor of scale."""
    return frame.resize((int(frame.width * scale), int(frame.height * scale)))


def save_gif(images, path, duration=None, fps=30):
    import imageio
    images = [np.asarray(image) for image in images]
    if fps is not None:
        imageio.mimsave(path, images, fps=fps)
    else:
        assert duration is not None
        imageio.mimsave(path, images, duration=duration)


def show_subsampled_frames(frames, n_show, figsize=(15, 3), as_canvas=True):
    indices = np.arange(len(frames))
    indices = np.linspace(0, len(frames) - 1, n_show, dtype=int)
    show_frames = [frames[i] for i in indices]
    if as_canvas:
        return concat_images(show_frames)
    else:
        show_grid_of_images(show_frames, n_cols=n_show, figsize=figsize, subtitles=indices)


def tensor_to_heatmap(x, scale=True, cmap="viridis", flip_vertically=False):
    import PIL
    
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    
    if scale:
        x = (x - x.min()) / (x.max() - x.min())
    
    cm = plt.get_cmap(cmap)
    if flip_vertically:
        x = np.flip(x, axis=0) # put low frequencies at the bottom in image
    x = cm(x)
    x = (x * 255).astype(np.uint8)
    if x.shape[-1] == 3:
        x = PIL.Image.fromarray(x, mode="RGB")
    elif x.shape[-1] == 4:
        x = PIL.Image.fromarray(x, mode="RGBA").convert("RGB")
    else:
        raise ValueError(f"Invalid shape {x.shape}")
    return x


def batch_tensor_to_heatmap(
        x, scale=True, cmap="viridis", flip_vertically=False, resize=None,
        concat=False,
    ):
    y = []
    for i in range(len(x)):
        h = tensor_to_heatmap(x[i], scale, cmap, flip_vertically)
        if resize is not None:
            h = h.resize(resize)
        y.append(h)
    if concat:
        y = concat_images_with_border(y)
    return y


def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)


def change_brightness(img, alpha):
    import PIL
    enhancer = PIL.ImageEnhance.Brightness(img)
    # to reduce brightness by 50%, use factor 0.5
    img = enhancer.enhance(alpha)
    return img


def draw_horizontal_lines(image, y_values, color=(255, 0, 0), colors=None, line_thickness=2):
    """
    Draw horizontal lines on a PIL image at specified Y positions.

    Args:
        image (PIL.Image.Image): The input PIL image.
        y_values (list or int): List of Y positions where lines will be drawn.
                               If a single integer is provided, a line will be drawn at that Y position.
        color (tuple): RGB color tuple (e.g., (255, 0, 0) for red).
        line_thickness (int): Thickness of the lines.

    Returns:
        PIL.Image.Image: The PIL image with the drawn lines.
    """
    image = image.copy()
    
    if isinstance(color, str):
        color = PIL.ImageColor.getcolor(color, "RGB")
        
    if colors is None:
        colors = [color] * len(y_values)
    else:
        if isinstance(colors[0], str):
            colors = [PIL.ImageColor.getcolor(c, "RGB") for c in colors]
    
    if isinstance(y_values, int):
        y_values = [y_values]
    
    # Create a drawing context on the image
    draw = PIL.ImageDraw.Draw(image)

    if isinstance(y_values, int):
        y_values = [y_values]

    for y, c in zip(y_values, colors):
        draw.line([(0, y), (image.width, y)], fill=c, width=line_thickness)

    return image


def draw_vertical_lines(image, x_values, color=(255, 0, 0), colors=None, line_thickness=2):
    """
    Draw vertical lines on a PIL image at specified X positions.

    Args:
        image (PIL.Image.Image): The input PIL image.
        x_values (list or int): List of X positions where lines will be drawn.
                               If a single integer is provided, a line will be drawn at that X position.
        color (tuple): RGB color tuple (e.g., (255, 0, 0) for red).
        line_thickness (int): Thickness of the lines.

    Returns:
        PIL.Image.Image: The PIL image with the drawn lines.
    """
    image = image.copy()
    
    if isinstance(color, str):
        color = PIL.ImageColor.getcolor(color, "RGB")
        
    if colors is None:
        colors = [color] * len(x_values)
    else:
        if isinstance(colors[0], str):
            colors = [PIL.ImageColor.getcolor(c, "RGB") for c in colors]
    
    if isinstance(x_values, int):
        x_values = [x_values]
    
    # Create a drawing context on the image
    draw = PIL.ImageDraw.Draw(image)

    if isinstance(x_values, int):
        x_values = [x_values]

    for x, c in zip(x_values, colors):
        draw.line([(x, 0), (x, image.height)], fill=c, width=line_thickness)

    return image


def show_arrow_on_image(image, start_loc, end_loc, color="red", thickness=3):
    """Draw a line on PIL image from start_loc to end_loc."""
    image = image.copy()
    color = get_predominant_color(color)

    # Create an instance of the ImageDraw class
    draw = ImageDraw.Draw(image)

    # Draw the bounding box on the image
    draw.line([start_loc, end_loc], fill=color, width=thickness)

    return image


def draw_arrow_on_image_cv2(image, start_loc, end_loc, color="red", thickness=2, both_ends=False):
    image = image.copy()
    image = np.asarray(image)
    if isinstance(color, str):
        color = PIL.ImageColor.getcolor(color, "RGB")
    image = cv2.arrowedLine(image, start_loc, end_loc, color, thickness)
    if both_ends:
        image = cv2.arrowedLine(image, end_loc, start_loc, color, thickness)
    return PIL.Image.fromarray(image)


def draw_arrow_with_text(image, start_loc, end_loc, text="", color="red", thickness=2, font_size=20, both_ends=False, delta=5):
    image = np.asarray(image)
    if isinstance(color, str):
        color = PIL.ImageColor.getcolor(color, "RGB")

    # Calculate the center point between start_loc and end_loc
    center_x = (start_loc[0] + end_loc[0]) // 2
    center_y = (start_loc[1] + end_loc[1]) // 2
    center_point = (center_x, center_y)

    # Draw the arrowed line
    image = cv2.arrowedLine(image, start_loc, end_loc, color, thickness)
    if both_ends:
        image = cv2.arrowedLine(image, end_loc, start_loc, color, thickness)

    # Create a PIL image from the NumPy array for drawing text
    image_with_text = Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(image_with_text)
    
    # Calculate the text size
    # font = PIL.ImageFont.truetype("arial.ttf", font_size)
    # This gives an error: "OSError: cannot open resource", as a hack, use the following
    text_width, text_height = draw.textsize(text)
    
    # Calculate the position to center the text
    text_x = center_x - (text_width // 2) - delta
    text_y = center_y - (text_height // 2)

    # Draw the text
    draw.text((text_x, text_y), text, color)

    return image_with_text


def draw_arrowed_line(image, start_loc, end_loc, color="red", thickness=2):
    """
    Draw an arrowed line on a PIL image from a starting point to an ending point.

    Args:
        image (PIL.Image.Image): The input PIL image.
        start_loc (tuple): Starting point (x, y) for the arrowed line.
        end_loc (tuple): Ending point (x, y) for the arrowed line.
        color (str): Color of the line (e.g., 'red', 'green', 'blue').
        thickness (int): Thickness of the line and arrowhead.

    Returns:
        PIL.Image.Image: The PIL image with the drawn arrowed line.
    """
    image = image.copy()
    if isinstance(color, str):
        color = PIL.ImageColor.getcolor(color, "RGB")
    
    
    # Create a drawing context on the image
    draw = ImageDraw.Draw(image)

    # Draw a line from start to end
    draw.line([start_loc, end_loc], fill=color, width=thickness)

    # Calculate arrowhead points
    arrow_size = 10  # Size of the arrowhead
    dx = end_loc[0] - start_loc[0]
    dy = end_loc[1] - start_loc[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    cos_theta = dx / length
    sin_theta = dy / length
    x1 = end_loc[0] - arrow_size * cos_theta
    y1 = end_loc[1] - arrow_size * sin_theta
    x2 = end_loc[0] - arrow_size * sin_theta
    y2 = end_loc[1] + arrow_size * cos_theta
    x3 = end_loc[0] + arrow_size * sin_theta
    y3 = end_loc[1] - arrow_size * cos_theta

    # Draw the arrowhead triangle
    draw.polygon([end_loc, (x1, y1), (x2, y2), (x3, y3)], fill=color)

    return image


def center_crop_to_fraction(image, frac=0.5):
    """Center crop an image to a fraction of its original size."""
    width, height = image.size
    new_width = int(width * frac)
    new_height = int(height * frac)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return image.crop((left, top, right, bottom))


def decord_load_frames(vr, frame_indices):
    if isinstance(frame_indices, int):
        frame_indices = [frame_indices]
    frames = vr.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def paste_mask_on_image(original_image, bounding_box, mask):
    """
    Paste a 2D mask onto the original image at the location specified by the bounding box.

    Parameters:
    - original_image (PIL.Image): The original image.
    - bounding_box (tuple): Bounding box coordinates (left, top, right, bottom).
    - mask (PIL.Image): The 2D mask.

    Returns:
    - PIL.Image: Image with the mask pasted on it.

    Example:
    ```
    original_image = Image.open('original.jpg')
    bounding_box = (100, 100, 200, 200)
    mask = Image.open('mask.png')
    result_image = paste_mask_on_image(original_image, bounding_box, mask)
    result_image.show()
    ```
    """
    # Create a copy of the original image to avoid modifying the input image
    result_image = original_image.copy()

    # Crop the mask to the size of the bounding box
    mask_cropped = mask.crop((0, 0, bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]))

    # Paste the cropped mask onto the original image at the specified location
    result_image.paste(mask_cropped, (bounding_box[0], bounding_box[1]))

    return result_image


def display_images_as_video_moviepy(image_list, fps=5, show=True):
    """
    Display a list of PIL images as a video in Jupyter Notebook using MoviePy.

    Parameters:
    - image_list (list): List of PIL images.
    - fps (int): Frames per second for the video.
    - show (bool): Whether to display the video in the notebook.

    Example:
    ```
    image_list = [Image.open('frame1.jpg'), Image.open('frame2.jpg'), ...]
    display_images_as_video_moviepy(image_list, fps=10)
    ```
    """
    from IPython.display import display
    from moviepy.editor import ImageSequenceClip

    image_list = list(map(np.asarray, image_list))
    clip = ImageSequenceClip(image_list, fps=fps)
    if show:
        display(clip.ipython_display(width=200))
    os.remove("__temp__.mp4")


def resize_height(img, H):
    w, h = img.size
    asp_ratio = w / h
    W = np.ceil(asp_ratio * H).astype(int)
    return img.resize((W, H))


def resize_width(img, W):
    w, h = img.size
    asp_ratio = w / h
    H = int(W / asp_ratio)
    return img.resize((W, H))


def resized_minor_side(img, size=256):
    H, W = img.size
    if H < W:
        H_new = size
        W_new = int(size * W / H)
        return img.resize((W_new, H_new))
    else:
        W_new = size
        H_new = int(size * H / W)
        return img.resize((W_new, H_new))


def brighten_image(img, alpha=1.2):
    enhancer = PIL.ImageEnhance.Brightness(img)
    img = enhancer.enhance(alpha)
    return img


def darken_image(img, alpha=0.8):
    enhancer = PIL.ImageEnhance.Brightness(img)
    img = enhancer.enhance(alpha)
    return img


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def show_temporal_tsne(
        tsne,
        timestamps=None,
        title="Feature projections over time",
        cmap='viridis',
        ax=None,
        fig=None,
        show=True,
        num_ticks=10,
        return_as_pil=False,
        dpi=100,
        label='Time (s)',
        figsize=(6, 4.5),
        xlim=None,
        ylim=None,
        s=None,
        tsne_kwargs=dict(),
        scatter=True,
        colorbar=True,
        marker="o",
        alpha=0.9,
        plot_label=None,
    ):

    if "method" not in tsne_kwargs:
        method = "tsne"
        tsne_kwargs["method"] = method
    else:
        method = tsne_kwargs["method"]
    title = f"{title} ({method.upper()})"

    if tsne.shape[1] > 2:
        tsne = reduce_dim(tsne, **tsne_kwargs)
    assert (tsne.shape[1] == 2), f"Invalid shape {tsne.shape}"

    if timestamps is None:
        timestamps = np.arange(len(tsne))

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    cmap = plt.get_cmap(cmap)
    if scatter:
        scatter = ax.scatter(
            tsne[:, 0], tsne[:, 1], c=np.arange(len(tsne)), cmap=cmap, s=s,
            edgecolor='k', linewidth=0.5, marker=marker, alpha=alpha, label=plot_label,
        )
    else:
        # Plot a line with circle markers
        # scatter = ax.plot(
        #     tsne[:, 0], tsne[:, 1], color="gray", alpha=0.4
        # )
        scatter = ax.scatter(
            tsne[:, 0], tsne[:, 1], c=np.arange(len(tsne)), cmap=cmap, s=s,
            edgecolor='k', linewidth=0.5, marker=marker, alpha=alpha, label=plot_label,
        )
        # Plot lines with averaged colors
        colors = cmap(np.linspace(0, 1, len(tsne)))
        for i in range(len(tsne) - 1):
            avg_color = (colors[i] + colors[i + 1]) / 2
            ax.plot(
                tsne[i:i + 2, 0], tsne[i:i + 2, 1], color=avg_color, linewidth=0.8, alpha=alpha,
            )
            ax.annotate(
                '', xy=tsne[i + 1], xytext=tsne[i],
                arrowprops=dict(
                    arrowstyle='->, head_width=0.3, head_length=0.5',
                    color=avg_color,
                    linewidth=0.9,
                    shrinkA=0,
                    shrinkB=0,
                ),
                alpha=0.5,
            )


    ax.grid(alpha=0.4)
    ax.set_title(f"{title}", fontsize=11)
    ax.set_xlabel("$z_{1}$")
    ax.set_ylabel("$z_{2}$")
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if plot_label is not None:
        ax.legend()

    # Create a colorbar
    if colorbar:
        cbar = fig.colorbar(
            scatter, ax=ax, label=label, location='bottom', fraction=0.1,
        )
    
        # Set custom ticks and labels on the colorbar
        ticks = np.linspace(0, len(tsne) - 1, num_ticks, dtype=int)
        tick_labels = np.round(timestamps[ticks], 1)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)

    plt.tight_layout()

    if show:
        plt.show()
    else:
        if return_as_pil:
            plt.tight_layout(pad=0.2)
            # fig.canvas.draw()
            # image = PIL.Image.frombytes(
            #     'RGB',
            #     fig.canvas.get_width_height(),
            #     fig.canvas.tostring_rgb(),
            # )
            # return image

            # Return as PIL Image without displaying the plt figure
            image = fig2img(fig)
            plt.close(fig)
            return image


def show_projections_with_labels(
        X, labels, title="",
        ax=None, fig=None, show=True, s=10, figsize=(6, 4), dpi=100,
        cmap="viridis", verbose=True, diff_markers=True, method="tsne", legend=True,
        alpha=0.8, legend_ncol=2, legend_outside=False,
):
    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        show = False
    labels = np.array(labels)

    # Apply projection
    if X.shape[1] == 2:
        Z = X
    else:
        if method == "tsne":
            Z = reduce_dim(X, method="tsne")
        elif method == "umap":
            Z = reduce_dim(X, method="umap")
        elif method == "pca":
            Z = reduce_dim(X, method="pca")
        else:
            raise ValueError(f"Unknown method: {method}")

    # Configure colors to use as per labels
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    colors = get_colors(n_labels, palette=cmap)
    if verbose:
        print("Number of unique labels:", len(unique_labels))
        # print("Colors: ", colors)
    

    # Set markers for each label
    if diff_markers:
        markers = ['o', 'X', '+', 'p', 's', '^', 'v', '<', '>', 'd', 'h']
        # Make them equal to the number of unique labels
        markers = markers * (n_labels // len(markers) + 1)
    else:
        markers = ['o'] * n_labels

    # Create a scatter plot
    for i, y in enumerate(unique_labels):
        indices = np.where(labels == y)
        ax.scatter(
            Z[indices, 0],
            Z[indices, 1],
            label=y,
            s=s,
            color=colors[i],
            marker=markers[i],
            alpha=alpha,
        )

    ax.grid(alpha=0.4)
    if title is not None and len(title):
        ax.set_title(f"{title}", fontsize=11)
    ax.set_xlabel("$z_{1}$")
    ax.set_ylabel("$z_{2}$")
    if legend:
        if legend_outside:
            ax.legend(ncol=legend_ncol, loc='upper right', bbox_to_anchor=(1.3, 1))
        else:
            ax.legend(ncol=legend_ncol)
    # ax.legend(title="Labels", loc='upper right', bbox_to_anchor=(1.3, 1))

    if show:
        plt.show()


def mark_keypoints(image, keypoints, color=(255, 255, 0), radius=1):
    """
    Marks keypoints on an image with a given color and radius.
    
    :param image: The input PIL image.
    :param keypoints: A list of (x, y) tuples representing the keypoints.
    :param color: The color to use for the keypoints (default: red).
    :param radius: The radius of the circle to draw for each keypoint (default: 5).
    :return: A new PIL image with the keypoints marked.
    """
    # Make a copy of the image to avoid modifying the original
    image_copy = image.copy()

    # Create a draw object to add graphical elements
    draw = ImageDraw.Draw(image_copy)

    # Loop through each keypoint and draw a circle
    for x, y in keypoints:
        # Draw a circle with the specified radius and color
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=color,
            width=2
        )

    return image_copy


def draw_line_on_image(image, x_coords, y_coords, color=(255, 255, 0), width=3):
    """
    Draws a line on an image given lists of x and y coordinates.
    
    :param image: The input PIL image.
    :param x_coords: List of x-coordinates for the line.
    :param y_coords: List of y-coordinates for the line.
    :param color: Color of the line in RGB (default is red).
    :param width: Width of the line (default is 3).
    :return: The PIL image with the line drawn.
    """
    image = image.copy()

    # Ensure the number of x and y coordinates are the same
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coords and y_coords must have the same length")

    # Create a draw object to draw on the image
    draw = ImageDraw.Draw(image)

    # Create a list of (x, y) coordinate tuples
    coordinates = list(zip(x_coords, y_coords))

    # Draw the line connecting the coordinates
    draw.line(coordinates, fill=color, width=width)

    return image


def add_binary_strip_vertically(
        image,
        binary_vector,
        strip_width=15,
        one_color="yellow",
        zero_color="gray",
):
    """
    Add a binary strip to the right side of an image.

    :param image: PIL Image to which the strip will be added.
    :param binary_vector: Binary vector of length 512 representing the strip.
    :param strip_width: Width of the strip to be added.
    :param one_color: Color for "1" pixels (default: red).
    :param zero_color: Color for "0" pixels (default: white).
    :return: New image with the binary strip added on the right side.
    """
    one_color = PIL.ImageColor.getrgb(one_color)
    zero_color = PIL.ImageColor.getrgb(zero_color)

    height = image.height
    if len(binary_vector) != height:
        raise ValueError("Binary vector must be of length 512")

    # Create a new strip with the specified width and 512 height
    strip = PIL.Image.new("RGB", (strip_width, height))

    # Fill the strip based on the binary vector
    pixels = strip.load()
    for i in range(height):
        color = one_color if binary_vector[i] == 1 else zero_color
        for w in range(strip_width):
            pixels[w, i] = color

    # Combine the original image with the new strip
    # new_image = PIL.Image.new("RGB", (image.width + strip_width, height))
    # new_image.paste(image, (0, 0))
    # new_image.paste(strip, (image.width, 0))
    new_image = image.copy()
    new_image.paste(strip, (image.width - strip_width, 0))

    return new_image


def add_binary_strip_horizontally(
    image,
    binary_vector,
    strip_height=15,
    one_color="limegreen",
    zero_color="gray",
):
    """
    Add a binary strip to the top of an image.

    :param image: PIL Image to which the strip will be added.
    :param binary_vector: Binary vector of length 512 representing the strip.
    :param strip_height: Height of the strip to be added.
    :param one_color: Color for "1" pixels, accepts color names or hex (default: red).
    :param zero_color: Color for "0" pixels, accepts color names or hex (default: white).
    :return: New image with the binary strip added at the top.
    """
    width = image.width
    if len(binary_vector) != width:
        raise ValueError("Binary vector must be of length 512")

    # Convert colors to RGB tuples
    one_color_rgb = PIL.ImageColor.getrgb(one_color)
    zero_color_rgb = PIL.ImageColor.getrgb(zero_color)

    # Create a new strip with the specified height and 512 width
    strip = PIL.Image.new("RGB", (width, strip_height))

    # Fill the strip based on the binary vector
    pixels = strip.load()
    for i in range(width):
        color = one_color_rgb if binary_vector[i] == 1 else zero_color_rgb
        for h in range(strip_height):
            pixels[i, h] = color

    # Combine the original image with the new strip
    # new_image = PIL.Image.new("RGB", (width, image.height + strip_height))
    # new_image.paste(strip, (0, 0))
    # new_image.paste(image, (0, strip_height))
    new_image = image.copy()
    new_image.paste(strip, (0, 0))

    return new_image


# Define a function to increase font sizes for a specific plot
def increase_font_sizes(ax, font_scale=1.6):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(item.get_fontsize() * font_scale)


def draw_multiple_boxes_on_image(image, boxes, colors=None, thickness=3):
    image = image.copy()
    if colors is None:
        colors = get_linspace_colors(num_colors=len(boxes), cmap_name="bwr")
    else:
        assert len(colors) == len(boxes)
    for box, color in zip(boxes, colors):
        image = add_box_on_image(image, box, color, thickness)
    return image



def mask_out_bbox(image, box):
    """
    Masks out a bounding box in an image by setting the pixel values to zero.

    Args:
        image (PIL.Image): The input image.
        box (list): The bounding box coordinates [xmin, ymin, xmax, ymax].
    """
    # Convert the image to a NumPy array
    image = np.array(image)

    # Extract the bounding box coordinates
    xmin, ymin, xmax, ymax = box

    # Mask out the bounding box by setting the pixel values to zero
    image[ymin:ymax, xmin:xmax] = 0

    # Convert the NumPy array back to a PIL image
    return Image.fromarray(image)


def sample_cmap(cmap='viridis', K=10):
    """
    Samples K colors from a given colormap.
    
    Args:
      cmap: The name of the matplotlib colormap to sample from. Defaults to 'viridis'.
      K: The number of colors to sample. Defaults to 10.
    
    Returns:
      A list of K RGB color tuples.
    """
    
    # Get the colormap object
    cmap_obj = plt.get_cmap(cmap)
    
    # Generate K evenly spaced values between 0 and 1
    values = np.linspace(0, 1, K)
    
    # Sample K colors from the colormap
    colors = [cmap_obj(value) for value in values]
    
    return colors


def cut_long_string(s, max_len=40):
    if len(s) > max_len:
        s = s[:max_len] + " ..."
    return s


import textwrap


def get_terminal_width():
    import shutil
    return shutil.get_terminal_size().columns


def wrap_text(text: str, max_length: int = 100) -> str:
    """
    Wraps a long string to the specified max_length for easier printing.

    Args:
        text (str): The input string to wrap.
        max_length (int): The maximum length of each line. Default is 80.

    Returns:
        str: The wrapped text with lines at most max_length long.
    """
    terminal_width = get_terminal_width()
    max_length = min(max_length, terminal_width)
    wrapped_text = textwrap.fill(text, width=max_length)
    return wrapped_text


def blank_image(color="black", size=(256, 256), border_color="black", border_width=2):
    """
    Creates a blank image with a specified color and size.

    Args:
        color (str): The color of the image. Default is 'white'.
        size (tuple): The size of the image in pixels. Default is (256, 256).

    Returns:
        PIL.Image: A blank image with the specified color and size.
    """
    image = Image.new("RGB", size, color)

    # Add border
    if border_width > 0:
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=border_color, width=border_width)
    
    return image


def insert_text_center(
        image,
        text,
        font_size=100,
        font_path="/users/piyush/.local/fonts/arial.ttf",
    ):
    assert os.path.exists(font_path), f"Font file not found at {font_path}"

    from PIL import ImageFont

    # Load font with the given font size
    font = ImageFont.truetype(font_path, font_size)
    
    # Initialize ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Calculate the bounding box of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Calculate x, y position to center the text
    x = (image.width - text_width) / 2
    y = (image.height - text_height) / 2
    
    # Draw the text onto the image
    draw.text((x, y), text, font=font, fill="black")  # You can change "black" to any color
    
    return image


from PIL import Image, ImageDraw, ImageColor

def add_shape_to_image(
    image,
    shape="circle",
    size=0.1,
    location=(0.5, 0.5),
    facecolor="red",
    edgecolor="red",
    edgethickness=2,
    check_bounds=True,
):
    """
    Adds a shape to an image at a specified location.

    Args:
        image (PIL.Image): The input image.
        shape (str): The shape to add. Can be 'circle', 'rectangle', or 'triangle'.
        size (float): The size of the shape as a fraction of the image size (min side).
        location (tuple): The location of the shape as a fraction of the image size.
        facecolor (str): The fill color of the shape.
        edgecolor (str): The edge color of the shape.
        edgethickness (int): The thickness of the edge.
        check_bounds (bool): Whether to check if the shape is within the image bounds.
    
    Returns:
        PIL.Image: The image with the shape added.
    """
    image = image.copy()

    # Get image dimensions
    width, height = image.size
    min_side = min(width, height)
    
    # Calculate center point in pixels
    center_x = int(location[0] * width)
    center_y = int(location[1] * height)
    
    # Initialize ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Calculate shape size in pixels
    shape_size = int(size * min_side)
    if shape == "circle":
        shape_size /= 2.
    
    # Check bounds if required
    if check_bounds:
        if shape == "circle":
            if not (0 <= center_x - shape_size < width and 0 <= center_x + shape_size < width and
                    0 <= center_y - shape_size < height and 0 <= center_y + shape_size < height):
                raise ValueError("The shape would be out of image bounds with the specified location and size.")
        
        elif shape == "rectangle":
            if not (0 <= center_x - shape_size // 2 < width and 0 <= center_x + shape_size // 2 < width and
                    0 <= center_y - shape_size // 2 < height and 0 <= center_y + shape_size // 2 < height):
                raise ValueError("The shape would be out of image bounds with the specified location and size.")
        
        elif shape == "triangle":
            if not (0 <= center_y - shape_size < height and
                    0 <= center_x - shape_size // 2 < width and 0 <= center_x + shape_size // 2 < width):
                raise ValueError("The shape would be out of image bounds with the specified location and size.")
    
    # Draw the specified shape
    if shape == "circle":
        # Draw circle as an ellipse
        bbox = [center_x - shape_size, center_y - shape_size, center_x + shape_size, center_y + shape_size]
        draw.ellipse(bbox, fill=facecolor, outline=edgecolor, width=edgethickness)
    
    elif shape == "rectangle":
        # Draw rectangle with min side length as specified size
        bbox = [center_x - shape_size // 2, center_y - shape_size // 2,
                center_x + shape_size // 2, center_y + shape_size // 2]
        draw.rectangle(bbox, fill=facecolor, outline=edgecolor, width=edgethickness)
    
    elif shape == "triangle":
        # Draw triangle
        points = [
            (center_x, center_y - shape_size),                # Top vertex
            (center_x - shape_size // 2, center_y + shape_size // 2),    # Bottom-left vertex
            (center_x + shape_size // 2, center_y + shape_size // 2),    # Bottom-right vertex
        ]
        draw.polygon(points, fill=facecolor, outline=edgecolor)
    
    else:
        raise ValueError("Shape not recognized. Use 'circle', 'rectangle', or 'triangle'.")
    
    return image


import math

def draw_star(
        center_x,
        center_y,
        outer_radius,
        inner_radius,
        num_points,
        draw_obj,
        facecolor,
        edgecolor,
        edgethickness,
    ):
    """
    Draws a star on the given `draw_obj`.

    Args:
        center_x (int): X-coordinate of the star's center.
        center_y (int): Y-coordinate of the star's center.
        outer_radius (int): Radius of the outer points of the star.
        inner_radius (int): Radius of the inner points of the star.
        num_points (int): Number of points the star has (must be >= 5).
        draw_obj (ImageDraw.Draw): The drawing object to draw the star.
        facecolor (tuple): RGBA color for the star's fill.
        edgecolor (tuple): RGBA color for the star's edge.
        edgethickness (int): Thickness of the star's edge.

    Returns:
        None
    """
    if num_points < 5:
        raise ValueError("Number of points must be at least 5 for a star.")

    # Generate points for the star
    points = []
    angle = 2 * math.pi / (2 * num_points)
    for i in range(2 * num_points):
        radius = outer_radius if i % 2 == 0 else inner_radius
        x = center_x + int(radius * math.sin(i * angle))
        y = center_y - int(radius * math.cos(i * angle))  # Invert y-axis for image coordinates
        points.append((x, y))

    # Draw the star
    draw_obj.polygon(points, fill=facecolor, outline=edgecolor)


def add_shape_to_image_with_opacity(
    image,
    shape="circle",
    size=0.1,
    location=(0.5, 0.5),
    facecolor="red",
    edgecolor="red",
    edgethickness=2,
    opacity=1.0,  # Opacity between 0 (transparent) and 1 (opaque)
    check_bounds=True,
):
    # Ensure opacity is within the valid range
    opacity = max(0.0, min(1.0, opacity))

    # Get image dimensions and calculate shape size
    width, height = image.size
    min_side = min(width, height)
    shape_size = int(size * min_side)
    center_x = int(location[0] * width)
    center_y = int(location[1] * height)
    if shape == "circle":
        shape_size /= 2.

    # Create an overlay image with transparency (RGBA)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    # Convert facecolor and edgecolor to RGBA with specified opacity
    if isinstance(facecolor, str):
        rgba_facecolor = (*ImageColor.getrgb(facecolor), int(255 * opacity))
    elif isinstance(facecolor, (np.ndarray, tuple, list)):
        assert len(facecolor) == 3, "RGB color must have 3 components"
        if np.max(facecolor) <= 1:
            facecolor = [int(255 * c) for c in facecolor]
        rgba_facecolor = (*list(facecolor), int(255 * opacity))
    else:
        raise ValueError("Facecolor must be a string or RGB tuple.")
    if isinstance(edgecolor, str):
        rgba_edgecolor = (*ImageColor.getrgb(edgecolor), int(255 * opacity))
    elif isinstance(edgecolor, (np.ndarray, tuple, list)):
        assert len(edgecolor) == 3, "RGB color must have 3 components"
        if np.max(edgecolor) <= 1:
            edgecolor = [int(255 * c) for c in edgecolor]
        rgba_edgecolor = (*list(edgecolor), int(255 * opacity))
    else:
        raise ValueError("Edgecolor must be a string or RGB tuple.")

    # Check bounds if required
    if check_bounds:
        if shape == "circle":
            if not (0 <= center_x - shape_size < width and 0 <= center_x + shape_size < width and
                    0 <= center_y - shape_size < height and 0 <= center_y + shape_size < height):
                raise ValueError("The shape would be out of image bounds with the specified location and size.")
        
        elif shape == "rectangle":
            if not (0 <= center_x - shape_size // 2 < width and 0 <= center_x + shape_size // 2 < width and
                    0 <= center_y - shape_size // 2 < height and 0 <= center_y + shape_size // 2 < height):
                raise ValueError("The shape would be out of image bounds with the specified location and size.")
        
        elif shape == "triangle":
            if not (0 <= center_y - shape_size < height and
                    0 <= center_x - shape_size // 2 < width and 0 <= center_x + shape_size // 2 < width):
                raise ValueError("The shape would be out of image bounds with the specified location and size.")
        
        elif shape == "star":
            if not (0 <= center_x - shape_size // 2 < width and 0 <= center_x + shape_size // 2 < width and
                    0 <= center_y - shape_size // 2 < height and 0 <= center_y + shape_size // 2 < height):
                raise ValueError("The shape would be out of image bounds with the specified location and size.")

    # Draw shapes on overlay using overlay_draw and rgba colors
    if shape == "circle":
        bbox = [center_x - shape_size, center_y - shape_size, center_x + shape_size, center_y + shape_size]
        overlay_draw.ellipse(bbox, fill=rgba_facecolor, outline=rgba_edgecolor, width=edgethickness)
    
    elif shape == "rectangle":
        bbox = [center_x - shape_size // 2, center_y - shape_size // 2,
                center_x + shape_size // 2, center_y + shape_size // 2]
        overlay_draw.rectangle(bbox, fill=rgba_facecolor, outline=rgba_edgecolor, width=edgethickness)
    
    elif shape == "triangle":
        points = [
            (center_x, center_y - shape_size // 2),
            (center_x - shape_size // 2, center_y + shape_size // 2),
            (center_x + shape_size // 2, center_y + shape_size // 2),
        ]
        overlay_draw.polygon(points, fill=rgba_facecolor, outline=rgba_edgecolor)
    
    elif shape == "star":
        # Define the star parameters
        outer_radius = shape_size // 2
        inner_radius = shape_size // 4
        num_points = 5  # Standard 5-point star

        # Draw the star
        draw_star(
            center_x, center_y, outer_radius, inner_radius,
            num_points, overlay_draw, rgba_facecolor,
            rgba_edgecolor, edgethickness,
        )

    # Composite the overlay onto the original image
    image = Image.alpha_composite(image.convert("RGBA"), overlay)

    return image.convert("RGB")  # Convert back to RGB if original was RGB


from PIL import Image, ImageDraw, ImageFont

def add_text_to_frames(
        image,
        text_numbers=["1", "2", "3"],
        position="top_left",
        font_size=50,
        font_color="red",
    ):
    """
    Adds text numbers to an image containing frames.

    Parameters:
        image (PIL.Image): Image containing frames.
        text_numbers (list): List of text numbers to add (e.g., ['1', '2', '3']).
        positions (list): List of positions for the text (e.g., ['top_left', 'top_right']).
        font_size (int): Font size of the text.
        output_path (str): Path to save the output image.
        font_color (str): Color of the font.

    Supported Positions:
        - 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    """
    image = image.copy()
    positions = [position] * len(text_numbers)

    # Open the image
    draw = ImageDraw.Draw(image)
    width, height = image.size
    frame_width = width // len(text_numbers)  # Assuming frames are equally spaced horizontally
    
    # Load a default font
    font = ImageFont.load_default(size=font_size)

    # Define offsets for each position
    offset_map = {
        'top_left': (10, 10),
        'top_right': (frame_width - 10, 10),
        'bottom_left': (10, height - 10),
        'bottom_right': (frame_width - 10, height - 10),
    }

    # Add text to each frame
    for i, (text, position) in enumerate(zip(text_numbers, positions)):
        x_offset = i * frame_width + 5
        x, y = offset_map[position]
        draw.text((x + x_offset, y), text, fill=font_color, font=font)

    return image


POSITIONS = [
    "top_left", "top_right", "bottom_left", "bottom_right", "center"
]


def add_texts_to_frames(
        frames,
        texts,
        fontsize=60,
        color="red",
        position="top_left",
    ):
    """
    Adds texts onto given frames.
    """
    assert len(frames) == len(texts), \
        "Number of frames and texts must be the same."
    assert position in POSITIONS, \
        f"Position must be one of {POSITIONS}."

    # Add text to each frame
    new_frames = []
    for frame, text in zip(frames, texts):
        frame = frame.copy()

        # Open the image
        draw = ImageDraw.Draw(frame)
        width, height = frame.size

        # Load a default font
        font = ImageFont.load_default(size=fontsize)

        # Define offsets for each position
        offset_map = {
            'top_left': (10, 10),
            'top_right': (width - 10, 10),
            'bottom_left': (10, height - 10),
            'bottom_right': (width - 10, height - 10),
            "center": (width // 2, height // 2),
        }

        # Add text to the frame
        x, y = offset_map[position]
        draw.text((x, y), text, fill=color, font=font)

        new_frames.append(frame)
    
    return new_frames


def add_border(image, color="red", thickness=10):
    """
    Adds border to an image without adding new pixels.
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle([0, 0, width - 1, height - 1], outline=color, width=thickness)
    return image


def visualize_dense_feature_norm(x, size=(224, 224), stitch=True):
    """
    Args:
        x (torch.Tensor): [F, H', W', D]
    """
    x = x.norm(dim=-1)
    x = torch.nn.functional.interpolate(
        x.unsqueeze(0), size=size, mode='bilinear', align_corners=False,
    ).squeeze(0)
    x = x.cpu().numpy()
    x = batch_tensor_to_heatmap(x)
    if stitch:
        x = concat_images_with_border(x)
    return x


def plot_confusion_matrix(x, cmap="viridis", title=None, show=False, return_as_pil=False):
    """
    Args:
        x (torch.Tensor): [C, C]
    """
    import seaborn as sns
    # x = x.cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    sns.heatmap(x, cmap=cmap, annot=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title(title)
    if show:
        plt.show()
    if return_as_pil:
        plt.tight_layout(pad=0.2)
        image = fig2img(fig)
        plt.close(fig)
        return image
    return fig


def show_text(text):
    from IPython.display import display, Markdown
    display(Markdown(text))


def show_text_with_color(text, color="red"):
    from IPython.display import display, Markdown
    display(Markdown(f"<span style='color: {color};'>{text}</span>"))