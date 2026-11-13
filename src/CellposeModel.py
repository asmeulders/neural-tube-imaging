import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, io
from cellpose.io import imread

class CellposeModel:
    def __init__(self, path="", gpu=True):
        self.model = models.CellposeModel(gpu=gpu)
        self.folder = path

    def set_folder(self, path):
        self.folder = path

    def eval(self, imgs):
        # Converts to nparray to avoid inputting PNG directly
        imgs = [np.asarray(img) for img in imgs]
        return self.model.eval(imgs)

    def eval_one(self, img):
        imgs = [np.asarray(img)]
        return self.model.eval(imgs)
