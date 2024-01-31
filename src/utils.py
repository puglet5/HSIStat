import logging
import os
import time
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
from sklearn.decomposition import PCA, FastICA
from spectral import SpyFile
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


def parse_image(data_array: npt.NDArray[np.float_], rot_angle=0):
    res_data_array = np.zeros(
        [data_array.shape[0], data_array.shape[1], data_array.shape[2]]
    )
    for i in range(data_array.shape[2]):
        res_data_array[:, :, i] = rotate_image(data_array[:, :, i], rot_angle)
    return res_data_array


def rotate_image(img: npt.NDArray[np.float_], angle=0):
    (h, w) = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (w, h))


def extract_pixels(data):
    q = data.reshape(-1, data.shape[2])
    df = DataFrame(q)
    df.columns = [f"band{i}" for i in range(1, 1 + data.shape[2])]
    return df


def reduce_dimensions(
    res_data_array: npt.NDArray[np.float_], dims: int, method, **kwargs
):
    reducer: PCA | FastICA = method(dims, **kwargs)
    df = extract_pixels(res_data_array)
    dt = reducer.fit_transform(df.iloc[:, :-1].values)

    q = pd.DataFrame(data=dt)
    q.columns = [f"PC-{i}" for i in range(1, dims + 1)]
    pca_images = np.zeros(
        shape=[res_data_array.shape[0], res_data_array.shape[1], dims]
    )
    for i in range(dims):
        arr: npt.NDArray[float_] = np.array(q.loc[:, f"PC-{i + 1}"].values.reshape(res_data_array.shape[0], res_data_array.shape[1]))  # type: ignore
        arr = arr - np.min(arr)
        arr = arr / np.max(arr)
        pca_images[:, :, i] = arr

    loadings = pd.DataFrame(reducer.components_.T, columns=q.columns)

    return pca_images, loadings


def log_exec_time(f):
    @wraps(f)
    def _wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = f(*args, **kwargs)
        print(f"{f.__name__}: {time.perf_counter() - start_time} s.")
        return result

    return _wrapper  # type:ignore


def radiometric_correction(
    data_array,
    white_array,
    dark_array,
) -> npt.NDArray[np.float_]:
    white_scaled: npt.NDArray[np.float_] = (
        white_array.sum(axis=0) / white_array.shape[0]
    )
    dark_scaled: npt.NDArray[np.float_] = dark_array.sum(axis=0) / dark_array.shape[0]

    return (data_array - dark_scaled) / (white_scaled - dark_scaled)


def load_hsi_image(path: str):
    image_number = path.split("/")[-1]
    data_path = f"{path}/capture/"
    img_dark = envi.open(
        f"{data_path}/DARKREF_{image_number}.hdr",
        f"{data_path}/DARKREF_{image_number}.raw",
    )
    img_white = envi.open(
        f"{data_path}/WHITEREF{image_number}.hdr"
        f"{data_path}/WHITEREF{image_number}.raw"
    )
    img_data = envi.open(
        f"{data_path}/{image_number}.hdr", f"{data_path}/{image_number}.raw"
    )

    assert isinstance(img_dark, BilFile)
    assert isinstance(img_white, BilFile)
    assert isinstance(img_data, BilFile)

    data_array = img_data.load()
    white_array = img_white.load()
    dark_array = img_dark.load()
    assert isinstance(img_dark, BilFile)

    return radiometric_correction(data_array, white_array, dark_array)


def load_png_image(path: str):
    return cv2.imread(f"{path}.png")


@dataclass
class Project:
    directory: str
    image_labels: list[str] = field(init=False)

    def __post_init__(self):
        if not os.path.exists(self.directory):
            self.image_labels = []
        else:
            self.image_labels = os.listdir(self.directory)
