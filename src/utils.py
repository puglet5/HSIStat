import logging
import os
import time
import xml.etree.ElementTree as ET
from copy import copy
from dataclasses import dataclass, field
from functools import cached_property, wraps
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar

import coloredlogs
import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt
import pandas as pd
import spectral.io.envi as envi
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from spectral.io.envi import BilFile

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
            d[x.tag] = []  # type:ignore
        d[x.tag].append(xml_to_dict(x, False))  # type:ignore
    return d


@dataclass
class Image:
    path: str
    png_image: list[float] = field(init=False)
    hsi_image: npt.NDArray[np.float_] = field(
        init=False, default_factory=lambda: np.array([])
    )
    hsi_image_loaded: bool = field(init=False, default=False)
    pca_calculated: bool = field(init=False, default=False)
    label: str = field(init=False)
    pca: PCA = field(init=False, default=PCA(n_components=10, svd_solver="arpack"))
    pca_data: npt.NDArray | None = field(init=False, default=None)
    pca_dimensions: int = field(init=False, default=0)
    raw_metadata: dict[str, str] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.label = self.path.split("/")[-1]
        self.load_png_image()
        self.parse_xml_metadata()

    @cached_property
    def histogram_data(self):
        data = self.hsi_image.mean(axis=1).flatten()
        data = data / np.max(data) * 255

        return data.tolist()

    @cached_property
    def integration_time(self):
        return self.raw_metadata["properties"]["dataset"][0]["key"][-1][  # type:ignore
            "_text"
        ]

    def parse_xml_metadata(self):
        xml_file = f"{self.path}/metadata/{self.label}.xml"
        with open(xml_file, "r") as f:
            xml = ET.fromstringlist(f.readlines())
            self.raw_metadata = xml_to_dict(xml)

    @property
    def pca_images(self):
        if self.pca_data is None:
            return

        return [
            cv2.cvtColor(self.pca_data[:, i], cv2.COLOR_GRAY2RGBA).flatten().tolist()
            for i in range(self.pca_data.shape[-1])
        ]

    def reduce_dimensions(self, dims: int):
        if self.pca_dimensions == dims:
            return

        self.pca_dimensions = dims

        reducer = self.pca = PCA(n_components=dims, svd_solver="arpack")
        reduced = reducer.fit_transform(self.hsi_image.reshape(-1, 204))
        for i in range(dims):
            minmax_scale(reduced[:, i], feature_range=(0, 1), copy=False)

        self.pca_data = reduced

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
        rgba_data = cv2.imread(f"{self.path}/{self.label}.png", cv2.IMREAD_UNCHANGED)

        bgra_data = cv2.cvtColor(rgba_data, cv2.COLOR_RGBA2BGRA)

        self.png_image = np.divide(bgra_data.flatten(), 255).tolist()

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


@dataclass
class Project:
    directory: str
    images: dict[str, Image] = field(init=False, default_factory=dict)
    current_image: tuple[str, Image] | None = field(init=False, default=None)

    def __post_init__(self):
        if not os.path.exists(self.directory):
            self.images = {}
        else:
            for label in os.listdir(self.directory):
                self.images[label] = Image(f"{self.directory}/{label}")

    def set_current_image(self, label: str):
        self.current_image = (label, self.images[label])
        return self.current_image
