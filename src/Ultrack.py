import napari
from ultrack import MainConfig, Tracker
from stardist.models import StarDist2D

class Ultrack:
    def __init__(self, model, folder=""):
        self.model = model
        self.folder = folder

    