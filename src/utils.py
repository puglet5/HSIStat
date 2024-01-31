import logging
import os
import time
import xml.etree.ElementTree as ET
from copy import copy
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Literal,
    NotRequired,
    ParamSpec,
    TypedDict,
    TypeVar,
)

import coloredlogs
import cv2
import dearpygui.dearpygui as dpg
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import spectral.io.envi as envi
from cv2.typing import MatLike
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.decomposition import PCA
from spectral.io.envi import BilFile

matplotlib.use("agg")
plt.ioff()
pd.set_option("future.no_silent_downcasting", True)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=Path(ROOT_DIR, "log/main.log"), filemode="a")
coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


def log_exec_time(f):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = f(*args, **kwargs)
        print(f"{f.__name__}: {time.perf_counter() - start_time} s.")
        return result

    return _wrapper  # type:ignore


def xml_to_dict(r: ET.Element, root=True):
    if root:
        return {r.tag: xml_to_dict(r, False)}
    d = copy(r.attrib)
    if r.text:
        d["_text"] = r.text
    for x in r.findall("./*"):
        if x.tag not in d:
            d[x.tag] = []
        d[x.tag].append(xml_to_dict(x, False))
    return d


@dataclass
class Image:
    path: str
    png_image: MatLike = field(init=False)
    hsi_image: npt.NDArray[np.float_] = field(
        init=False, default_factory=lambda: np.array([])
    )
    hsi_image_loaded: bool = False
    label: str = field(init=False)
    pca: PCA = field(init=False, default=PCA(n_components=10))
    metadata: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.label = self.path.split("/")[-1]
        self.load_png_image()
        self.parse_xml_metadata()

    @property
    def histogram_data(self):
        data = self.hsi_image.mean(axis=1).flatten()
        data = data / np.max(data) * 255

        return data.tolist()

    def parse_xml_metadata(self):
        xml_file = f"{self.path}/metadata/{self.label}.xml"
        with open(xml_file, "r") as f:
            xml = ET.fromstringlist(f.readlines())
            self.metadata = xml_to_dict(xml)

    def reduce_dimensions(self, dims: int):
        reducer: PCA = PCA(n_components=10)
        df = self.extract_pixels(self.hsi_image)
        dt = reducer.fit_transform(df.iloc[:, :-1].values)

        q = pd.DataFrame(data=dt)
        q.columns = [f"PC-{i}" for i in range(1, dims + 1)]
        pca_images = np.zeros(
            shape=[self.hsi_image.shape[0], self.hsi_image.shape[1], dims]
        )
        for i in range(dims):
            arr: npt.NDArray[float_] = np.array(q.loc[:, f"PC-{i + 1}"].values.reshape(res_data_array.shape[0], res_data_array.shape[1]))  # type: ignore
            arr = arr - np.min(arr)
            arr = arr / np.max(arr)
            pca_images[:, :, i] = arr

        loadings = pd.DataFrame(reducer.components_.T, columns=q.columns)

        return pca_images, loadings

    def load_hsi_image(self):
        if self.hsi_image_loaded:
            return

        data_path = f"{self.path}/capture"
        img_dark = envi.open(
            f"{data_path}/DARKREF_{self.label}.hdr",
            f"{data_path}/DARKREF_{self.label}.raw",
        )
        img_white = envi.open(
            f"{data_path}/WHITEREF_{self.label}.hdr",
            f"{data_path}/WHITEREF_{self.label}.raw",
        )
        img_data = envi.open(
            f"{data_path}/{self.label}.hdr", f"{data_path}/{self.label}.raw"
        )

        assert isinstance(img_dark, BilFile)
        assert isinstance(img_white, BilFile)
        assert isinstance(img_data, BilFile)

        data_array = img_data.load()
        white_array = img_white.load()
        dark_array = img_dark.load()

        self.hsi_image = self.radiometric_correction(
            data_array, white_array, dark_array
        )

        self.hsi_image_loaded = True

    def load_png_image(self):
        self.png_image = cv2.imread(f"{self.path}/{self.label}.png")

    def radiometric_correction(
        self,
        data_array,
        white_array,
        dark_array,
    ) -> npt.NDArray[np.float_]:
        white_scaled: npt.NDArray[np.float_] = (
            white_array.sum(axis=0) / white_array.shape[0]
        )
        dark_scaled: npt.NDArray[np.float_] = (
            dark_array.sum(axis=0) / dark_array.shape[0]
        )

        return (data_array - dark_scaled) / (white_scaled - dark_scaled)

    def extract_pixels(self, data):
        q = data.reshape(-1, data.shape[2])
        df = DataFrame(q)
        df.columns = [f"band{i}" for i in range(1, 1 + data.shape[2])]
        return df


@dataclass
class Project:
    directory: str
    images: dict[str, Image] = field(init=False, default_factory=dict)
    current_image: str | None = field(init=False, default=None)

    def __post_init__(self):
        if not os.path.exists(self.directory):
            self.images = {}
        else:
            for label in os.listdir(self.directory):
                self.images[label] = Image(f"{self.directory}/{label}")
