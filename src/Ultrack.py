import napari
import numpy as np
from ultrack import MainConfig, track, to_tracks_layer, tracks_to_zarr
from stardist.models import StarDist2D

from csbdeep.utils import normalize
from ultrack.utils import labels_to_contours

class Ultrack:
    def __init__(self, folder=""):
        # greyscale images
        self.model = StarDist2D.from_pretrained('2D_versatile_fluo')
        self.folder = folder
        self.config = MainConfig()
        self.config.data_config.working_dir = "./ultrack_run"
        self.config.segmentation_config.min_area = 50          # px, adjust to your nuclei
        self.config.linking_config.max_distance = 30           # px per frame, max expected movement
        self.config.tracking_config.disappear_weight = -1.0    # tune these for your data
        self.config.tracking_config.appear_weight = -1.0
        self.config.tracking_config.division_weight = -0.1


    def predict(self, img: np.ndarray):
        return self.model.predict_instances(normalize(img))

    def track(self, labels):
        foreground, contours = labels_to_contours(labels, sigma=2.0)
        track(foreground=foreground, contours=contours, config=self.config)

        tracks_df, graph = to_tracks_layer(self.config)
        tracked_labels = tracks_to_zarr(self.config, tracks_df)

        return tracks_df, graph, tracked_labels


# stack shape, frame interval?