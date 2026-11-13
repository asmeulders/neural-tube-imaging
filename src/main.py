import matplotlib.pyplot as plt
from cellpose import plot
from pathlib import Path
import os
from PIL import Image
import numpy as np


from CellposeModel import CellposeModel
import OpenCV

def cellpose(data_dir):
    model = CellposeModel(path=os.path.join(data_dir, "neural_tube_data_raw_nucleusc1_membranec2"))
    
    image_paths = sorted([
        p for p in Path(model.folder).iterdir() if p.suffix.lower() == ".png"
    ])

    imgs = [Image.open(p) for p in image_paths]
    img = np.asarray(imgs[1])

    masks, flows, styles = model.eval_one(img=imgs[0])

    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img, masks[0], flows[0][0])
    plt.tight_layout()
    plt.show()


def canny(data_dir):
    dir = os.path.join(data_dir, "neural_tube_data_raw_nucleusc1_membranec2")

    image_paths = sorted([
            p for p in Path(dir).iterdir() if p.suffix.lower() == ".png"
        ])
    imgs = [Image.open(p) for p in image_paths]
    img = np.asarray(imgs[1])
    
    edge = OpenCV.canny(img=img)

    plt.imshow(edge, cmap="gray")
    plt.show()

def main():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    data_dir = os.path.join(project_dir, "data")

    canny(data_dir)


if __name__ == "__main__":
    main()