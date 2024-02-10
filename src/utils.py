import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from copy import copy
from enum import Enum
from functools import cached_property, wraps
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

import coloredlogs
import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt
from attrs import define, field
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from spectral.io import envi
from spectral.io.envi import BilFile

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=Path(ROOT_DIR, "log/main.log"), filemode="a")
coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


def log_exec_time(f: Callable[P, T]) -> Callable[P, T]:  # type:ignore
    @wraps(f)
    def _wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = f(*args, **kwargs)
        print(f"{f.__name__}: {time.perf_counter() - start_time} s.")
        return result

    return _wrapper  # type:ignore


def show_loading_indicator():
    dpg.show_item("loading_indicator")


def hide_loading_indicator():
    if dpg.is_item_shown("loading_indicator"):
        dpg.hide_item("loading_indicator")


def loading_indicator(f: Callable[P, T], message: str) -> Callable[P, T]:  # type:ignore
    @wraps(f)
    def _wrapper(*args, **kwargs):
        dpg.configure_item("loading_indicator_message", label=message.center(30))
        threading.Timer(0.1, show_loading_indicator).start()
        result = f(*args, **kwargs)
        threading.Timer(0.1, hide_loading_indicator).start()
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


def rotate_image(img, rot_angle=0):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rot_angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (w, h))


class ImageType(Enum):
    HSI = 1
    PNG = 2
    CHANNEL = 3
    PCA = 4
    OTHER = 5


@define
class Image:
    path: str
    png_image: npt.NDArray[np.float_] = field(init=False)
    hsi_image: npt.NDArray[np.float_] = field(init=False, factory=lambda: np.array([]))
    hsi_image_loaded: bool = field(init=False, default=False)
    pca_calculated: bool = field(init=False, default=False)
    label: str = field(init=False)
    pca: PCA = field(init=False, default=PCA(n_components=10, svd_solver="arpack"))
    pca_data: npt.NDArray | None = field(init=False, default=None)
    pca_dimensions: int = field(init=False, default=0)
    raw_metadata: dict[str, Any] = field(init=False, factory=dict)

    def __attrs_post_init__(self):
        self.label = self.path.split("/")[-1]
        self.load_png_image()
        self.parse_xml_metadata()

    def histogram_data(
        self, source: npt.NDArray[np.float_], axis, source_type: ImageType
    ):
        if source_type != ImageType.HSI:
            source = source[:, :, :-1]
        data = source.mean(axis=axis).flatten()
        data = data / np.max(data) * 255

        return data.tolist()

    @log_exec_time
    def channel_images(self):
        if self.hsi_image is None:
            return None

        images = np.array(
            [
                self._channel_to_image(self.hsi_image[:, :, i])
                for i in range(self.hsi_image.shape[-1])
            ]
        )

        return images

    def _channel_to_image(self, to_convert: npt.NDArray[np.float_]):
        return rotate_image(cv2.cvtColor(to_convert, cv2.COLOR_GRAY2RGBA), -90)

    @cached_property
    def integration_time(self):
        return self.raw_metadata["properties"]["dataset"][0]["key"][-1]["_text"]

    @cached_property
    def datetime(self):
        return self.raw_metadata["properties"]["dataset"][0]["key"][0]["_text"]

    def parse_xml_metadata(self):
        xml_file = f"{self.path}/metadata/{self.label}.xml"
        with open(xml_file, "r", encoding="utf8") as f:
            xml = ET.fromstringlist(f.readlines())
            self.raw_metadata = xml_to_dict(xml)

    def _pca_to_image(self, to_convert, apply_clahe: bool):
        img = to_convert.reshape(512, 512, 1)

        if apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = (img * 255).astype(np.uint8)
            img = clahe.apply(img)
            img = img.astype(np.float32) / 255

        img = rotate_image(
            cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA),
            -90,
        )

        return img

    def pca_images(self, apply_clahe):
        if self.pca_data is None:
            return None

        return np.array(
            [
                self._pca_to_image(self.pca_data[:, i], apply_clahe)
                for i in range(self.pca_data.shape[-1])
            ]
        )

    def reduce_hsi_dimensions(self, dims: int):
        if self.pca_dimensions == dims:
            return

        reducer = self.pca = PCA(n_components=dims, svd_solver="arpack")
        reduced = reducer.fit_transform(self.hsi_image.reshape(-1, 204))
        for i in range(dims):
            minmax_scale(reduced[:, i], feature_range=(0, 1), copy=False)

        self.pca_dimensions = dims
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

        self.png_image = np.divide(bgra_data, 255)

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


@define
class Project:
    catalog: str
    images: dict[str, Image] = field(init=False, factory=dict)
    current_image: tuple[str, Image] | None = field(init=False, default=None)
    image_dirs: list[str] | None = field(init=False, default=None)

    def __attrs_post_init__(self):
        if not os.path.exists(self.catalog):
            self.images = {}
        else:
            self.validate_catalog()
            assert self.image_dirs is not None
            for directory in self.image_dirs:
                self.images[directory] = Image(f"{self.catalog}/{directory}")

    def validate_catalog(self):
        catalog_contents = os.listdir(self.catalog)
        possible_image_dirs = [
            i for i in catalog_contents if os.path.isdir(f"{self.catalog}/{i}")
        ]
        if not any(possible_image_dirs):
            raise ValueError
        for directory in possible_image_dirs:
            if not re.match(r"\d", directory):
                raise ValueError

        self.image_dirs = possible_image_dirs

    def set_current_image(self, label: str):
        self.current_image = (label, self.images[label])
        return self.current_image
