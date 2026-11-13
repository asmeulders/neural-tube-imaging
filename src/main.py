import matplotlib.pyplot as plt
from cellpose import plot
from pathlib import Path
import os
import re
from PIL import Image
import numpy as np
import napari
from skimage.io import imread

from ultrack.utils.array import array_apply
from napari.utils.notebook_display import nbscreenshot
from stardist.plot import render_label

from CellposeModel import CellposeModel
import OpenCV
from Ultrack import Ultrack

def get_imgs(dir, extension) -> list:
    image_paths = sorted([
        p for p in Path(dir).iterdir() if p.suffix.lower() == extension
    ])

    return [Image.open(p) for p in image_paths]

def natural_key(p):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)]

def cellpose(data_dir):
    model = CellposeModel(path=os.path.join(data_dir, "neural_tube_data_raw_nucleusc1_membranec2"))

    imgs = get_imgs(model.folder, ".png")
    img = np.asarray(imgs[1])

    masks, flows, styles = model.eval_one(img=imgs[0])

    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img, masks[0], flows[0][0])
    plt.tight_layout()
    plt.show()


def canny(data_dir):
    dir = os.path.join(data_dir, "neural_tube_data_raw_nucleusc1_membranec2")

    imgs = get_imgs(dir, ".png")
    img = np.asarray(imgs[1])
    
    edge = OpenCV.canny(img=img)

    plt.imshow(edge, cmap="gray")
    plt.show()

def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    data_dir = os.path.join(project_dir, "data")

    model = Ultrack(os.path.join(data_dir, 'neural_tube_data_raw', 'nucleus'))


    files = sorted(Path(model.folder).glob("*.png"))
    files = sorted(files, key=natural_key)

    stack = np.stack([imread(f) for f in files])

    labels = np.zeros(stack.shape, dtype=np.int32)
    for t in range(stack.shape[0]):
        labels[t], _ = model.predict(stack[t])

    tracks_df, graph, tracked_labels = model.track(labels)

    viewer = napari.Viewer()
    lo, hi = np.percentile(stack, (1, 99.8)) # from normalization, default values from normalize in predict
    viewer.add_image(stack, name="raw", contrast_limits=(lo, hi))
    viewer.add_labels(labels, name="stardist (per-frame IDs)")
    viewer.add_labels(tracked_labels, name="tracked (consistent IDs)")
    viewer.add_tracks(
        tracks_df[["track_id", "t", "y", "x"]].to_numpy(),
        graph=graph,
        name="tracks",
    )
    napari.run()

    # plt.subplot(1,2,1)
    # plt.imshow(img, cmap="gray")
    # plt.axis("off")
    # plt.title("input image")

    # plt.subplot(1,2,2)
    # plt.imshow(render_label(labels, img=img))
    # plt.axis("off")
    # plt.title("prediction + input overlay")

    # plt.show()


if __name__ == "__main__":
    main()