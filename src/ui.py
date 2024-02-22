import uuid
from functools import partial
from typing import Literal

import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt
from attrs import define, field
from sklearn.decomposition import PCA, FastICA

from src.utils import (
    ImageType,
    Project,
    loading_indicator,
    log_exec_time,
    logger,
    rotate_image,
)

TOOLTIP_DELAY_SEC = 0.1
LABEL_PAD = 23


@define
class UI:
    project: Project = field(init=False)
    pca_images: npt.NDArray | None = field(init=False, default=None)
    channel_images: npt.NDArray | None = field(init=False, default=None)
    window_tag: Literal["primary_window"] = field(init=False, default="primary_window")
    gallery_tag: Literal["image_gallery"] = field(init=False, default="image_gallery")
    image_gallery_n_columns: Literal[9] = 9
    sidebar_width: Literal[350] = 350
    global_theme: int = field(init=False, default=0)
    button_theme: int = field(init=False, default=0)
    active_button_theme: int = field(init=False, default=0)
    normal_button_theme: int = field(init=False, default=0)

    def __attrs_post_init__(self):
        dpg.create_context()
        dpg.create_viewport(title="hsistat", width=800, height=600, vsync=True)
        dpg.configure_app(wait_for_input=False)
        self.setup_themes()
        self.bind_themes()
        self.setup_handler_registries()
        self.setup_layout()
        self.bind_item_handlers()

    def start(self, dev=False):
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_vsync(True)
        dpg.set_primary_window(self.window_tag, True)
        try:
            if dev:
                dpg.set_frame_callback(1, self.setup_dev)
            dpg.start_dearpygui()
        except Exception as e:
            logger.fatal(e)
        finally:
            self.stop()

    def stop(self):
        dpg.stop_dearpygui()
        dpg.destroy_context()

    def setup_themes(self):
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_ButtonTextAlign, 0.5, category=dpg.mvThemeCat_Core
                )

            with dpg.theme_component(item_type=dpg.mvHistogramSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    (20, 119, 200, 255),
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_Fill,
                    (20, 119, 200, 255),
                    category=dpg.mvThemeCat_Plots,
                )

        with dpg.theme() as self.button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button,
                    (37, 37, 38, -255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive,
                    (37, 37, 38, -255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    (37, 37, 38, -255),
                    category=dpg.mvThemeCat_Core,
                )

        with dpg.theme() as self.active_button_theme:
            with dpg.theme_component(dpg.mvImageButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button,
                    (20, 119, 200, 255),
                    category=dpg.mvThemeCat_Core,
                )

        with dpg.theme() as self.normal_button_theme:
            with dpg.theme_component(dpg.mvImageButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button,
                    (52, 52, 55, 255),
                    category=dpg.mvThemeCat_Core,
                )

    def update_pca_component_spec(self, _sender, data):
        dpg.configure_item("pca_slider", max_value=data)
        current_pca_component = dpg.get_value("pca_slider")
        if current_pca_component > data:
            dpg.set_value("pca_slider", data)

        if not hasattr(self, "project"):
            return

        if self.project.current_image is not None:
            self.update_pca_images()

    def bind_themes(self):
        dpg.bind_theme(self.global_theme)

    def populate_image_gallery(self):
        if not self.project.images:
            dpg.show_item("invalid_directory_message")
        else:
            dpg.hide_item("invalid_directory_message")

        if dpg.does_item_exist("image_gallery_wrapper"):
            dpg.delete_item("image_gallery_wrapper")

        with dpg.group(tag="image_gallery_wrapper", parent=self.gallery_tag):
            for i in range(
                len(self.project.images.keys()) // self.image_gallery_n_columns + 1
            ):
                dpg.add_group(
                    tag=f"image_gallery_row_{i}",
                    horizontal=True,
                )

        for i, (label, image) in enumerate(self.project.images.items()):
            if not dpg.does_item_exist(f"image_{label}"):
                with dpg.texture_registry():
                    dpg.add_static_texture(
                        512,
                        512,
                        image.png_image.flatten().tolist(),
                        tag=f"image_{label}",
                    )

            with dpg.group(
                parent=f"image_gallery_row_{i//self.image_gallery_n_columns}",
                width=100,
                tag=f"image_{label}_wrapper",
            ):
                dpg.add_image_button(
                    f"image_{label}",
                    width=-1,
                    height=100,
                    tag=f"image_{label}_button",
                    user_data=label,
                    callback=lambda s, d: self.image_select_callback(s, d),
                )
                dpg.add_button(label=label, indent=2)
                dpg.bind_item_theme(dpg.last_item(), self.button_theme)
                with dpg.tooltip(
                    parent=f"image_{label}_button", delay=TOOLTIP_DELAY_SEC
                ):
                    dpg.add_text(
                        f"Integration time: {image.integration_time} ms\nDate: {image.datetime}"
                    )

    def setup_dev(self):
        self.project = Project("/run/media/puglet5/HP P600/IH/stripes")
        self.populate_image_gallery()
        self.window_resize_callback()

    def directory_picker_callback(self, _sender, data):
        dpg.set_value("project_directory", data["file_path_name"])
        self.setup_project()

    def setup_project(self):
        if hasattr(self, "project"):
            del self.project

        dpg.hide_item("project_directory_error_message")

        try:
            self.project = Project(dpg.get_value("project_directory"))
        except ValueError:
            dpg.show_item("project_directory_error_message")
            return

        self.populate_image_gallery()
        self.window_resize_callback()
        self.collapsible_clicked_callback()

    def save_pca_image(self):
        if not isinstance(self.pca_images, np.ndarray):
            return []
        if self.project.current_image is None:
            return

        name, curr_image = self.project.current_image

        pca_component_n = dpg.get_value("pca_slider") - 1

        image_data = self.pca_images[pca_component_n] * [255, 255, 255]

        image_data = image_data.flatten().astype(int)
        filename = f"{self.project.catalog}/{name}_PCA_{pca_component_n}.png"

        dpg.save_image(filename, 512, 512, image_data)

    def save_channel_image(self):
        if not isinstance(self.channel_images, np.ndarray):
            return []
        if self.project.current_image is None:
            return

        name, curr_image = self.project.current_image

        channel_n = dpg.get_value("channel_slider") - 1

        image_data = self.channel_images[channel_n] * [255, 255, 255]

        image_data = image_data.flatten().astype(int)
        filename = f"{self.project.catalog}/{name}_channel_{channel_n}.png"

        dpg.save_image(filename, 512, 512, image_data)

    def save_all_shown_images(self):
        self.save_channel_image()
        self.save_pca_image()

    def setup_layout(self):
        with dpg.window(
            label="hsistat",
            tag=self.window_tag,
            horizontal_scrollbar=False,
            on_close=self.stop,
            no_scrollbar=True,
            autosize=True,
            min_size=[160, 90],
        ):
            with dpg.menu_bar(tag="menu_bar"):
                with dpg.menu(label="File"):
                    dpg.add_menu_item(
                        label="Open new catalog",
                        shortcut="(Ctrl+O)",
                        callback=lambda: dpg.show_item("project_directory_picker"),
                    )
                    dpg.add_menu_item(
                        label="Save shown images",
                        shortcut="(Ctrl+S)",
                        callback=lambda s, d: self.save_all_shown_images(),
                    )
                    dpg.add_menu_item(label="Quit", shortcut="(Ctrl+Q)")

                with dpg.menu(label="Edit"):
                    dpg.add_menu_item(
                        label="Preferences",
                        shortcut="(Ctrl+,)",
                    )

                with dpg.menu(label="Window"):
                    dpg.add_menu_item(
                        label="Wait For Input",
                        check=True,
                        tag="wait_for_input_menu",
                        shortcut="(Ctrl+Shift+Alt+W)",
                        callback=lambda s, a: dpg.configure_app(wait_for_input=a),
                    )
                    dpg.add_menu_item(
                        label="Toggle Fullscreen",
                        shortcut="(F11)",
                        callback=lambda: dpg.toggle_viewport_fullscreen(),
                    )
                with dpg.menu(label="Tools"):
                    with dpg.menu(label="Developer"):
                        dpg.add_menu_item(
                            label="Show About",
                            callback=lambda: dpg.show_tool(dpg.mvTool_About),
                        )
                        dpg.add_menu_item(
                            label="Show Metrics",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Metrics),
                            shortcut="(Ctrl+Shift+Alt+M)",
                        )
                        dpg.add_menu_item(
                            label="Show Documentation",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Doc),
                        )
                        dpg.add_menu_item(
                            label="Show Debug",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Debug),
                        )
                        dpg.add_menu_item(
                            label="Show Style Editor",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Style),
                        )
                        dpg.add_menu_item(
                            label="Show Font Manager",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Font),
                        )
                        dpg.add_menu_item(
                            label="Show Item Registry",
                            callback=lambda: dpg.show_tool(dpg.mvTool_ItemRegistry),
                        )

            with dpg.group(horizontal=True):
                with dpg.child_window(
                    border=False,
                    width=self.sidebar_width,
                    tag="sidebar",
                    no_scrollbar=True,
                ):
                    with dpg.child_window(
                        label="Project",
                        width=-1,
                        height=200,
                        menubar=True,
                        no_scrollbar=True,
                    ):
                        with dpg.menu_bar():
                            with dpg.menu(label="Project", enabled=False):
                                pass
                        with dpg.group(tag="project_controls", horizontal=False):
                            with dpg.group(horizontal=True):
                                dpg.add_text("Catalog directory".rjust(LABEL_PAD))
                                dpg.add_input_text(
                                    tag="project_directory",
                                    width=100,
                                    callback=lambda s, d: self.setup_project(),
                                    on_enter=True,
                                )
                                dpg.add_button(
                                    label="Browse",
                                    width=-1,
                                    callback=lambda: dpg.show_item(
                                        "project_directory_picker"
                                    ),
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text(
                                    default_value="Chosen directory is not a catalog!".rjust(
                                        LABEL_PAD
                                    ),
                                    tag="project_directory_error_message",
                                    show=False,
                                    color=(200, 20, 20, 255),
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Load all images".rjust(LABEL_PAD))
                                dpg.add_checkbox(default_value=True)

                            with dpg.group(horizontal=True):
                                dpg.add_text("Rotate images".rjust(LABEL_PAD))
                                dpg.add_combo(
                                    items=["0", "90", "-90", "180"],
                                    default_value="0",
                                    width=-1,
                                    tag="rotate_images",
                                    callback=self.rotate_images,
                                )

                    with dpg.child_window(
                        label="Images",
                        width=-1,
                        height=200,
                        menubar=True,
                        no_scrollbar=True,
                    ):
                        with dpg.menu_bar():
                            with dpg.menu(label="Images", enabled=False):
                                pass
                        with dpg.group(tag="image_controls", horizontal=False):
                            with dpg.group(horizontal=True):
                                dpg.add_text("Histogram source".rjust(LABEL_PAD))
                                dpg.add_combo(
                                    items=["HSI", "PNG", "Channel", "PCA"],
                                    default_value="HSI",
                                    tag="histogram_source_combo",
                                    width=-1,
                                    callback=lambda s, d: self.update_histogram_plot(),
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Bin count".rjust(LABEL_PAD))
                                dpg.add_input_int(
                                    min_clamped=True,
                                    min_value=10,
                                    max_clamped=True,
                                    max_value=1000,
                                    width=-1,
                                    tag="histogram_bins",
                                    callback=lambda s, d: self.update_histogram_plot(),
                                    on_enter=True,
                                    default_value=255,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Histogram average over".rjust(LABEL_PAD))
                                dpg.add_combo(
                                    items=["Columns", "Rows"],
                                    default_value="Columns",
                                    tag="histogram_axis_combo",
                                    width=-1,
                                    callback=lambda s, d: self.update_histogram_plot(),
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text(
                                    default_value="Shown channel".rjust(LABEL_PAD)
                                )
                                dpg.add_slider_int(
                                    tag="channel_slider",
                                    callback=self.channel_slider_callback,
                                    min_value=1,
                                    max_value=204,
                                    clamped=True,
                                    default_value=1,
                                    width=-1,
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_text(default_value="".rjust(LABEL_PAD))
                                dpg.add_button(
                                    label="Save channel image",
                                    width=-1,
                                    callback=lambda s, d: self.save_channel_image(),
                                )

                    with dpg.child_window(
                        label="PCA",
                        width=-1,
                        height=250,
                        menubar=True,
                        no_scrollbar=True,
                    ):
                        with dpg.menu_bar():
                            with dpg.menu(label="PCA", enabled=False):
                                pass
                        with dpg.group(horizontal=True):
                            dpg.add_text("Reducer".rjust(LABEL_PAD))
                            dpg.add_combo(
                                items=["PCA", "ICA", "PCA->ICA"],
                                default_value="PCA",
                                width=-1,
                                tag="image_reducer",
                                callback=self.update_pca_images,
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text("Method".rjust(LABEL_PAD))
                            dpg.add_combo(
                                items=["auto", "full", "randomized"],
                                default_value="auto",
                                width=-1,
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text(
                                default_value="Number of components".rjust(LABEL_PAD)
                            )
                            dpg.add_input_int(
                                tag="pca_n_components",
                                default_value=10,
                                callback=self.update_pca_component_spec,
                                min_value=3,
                                max_value=20,
                                min_clamped=True,
                                max_clamped=True,
                                on_enter=True,
                                width=-1,
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text(
                                default_value="Shown component".rjust(LABEL_PAD)
                            )
                            dpg.add_slider_int(
                                tag="pca_slider",
                                callback=self.pca_component_slider_callback,
                                min_value=1,
                                max_value=10,
                                clamped=True,
                                default_value=1,
                                width=-1,
                            )

                        with dpg.group(horizontal=True):
                            dpg.add_text(default_value="Apply CLAHE".rjust(LABEL_PAD))
                            dpg.add_checkbox(
                                tag="apply_clahe_checkbox",
                                default_value=False,
                                callback=lambda s, d: self.update_pca_images(),
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text(default_value="".rjust(LABEL_PAD))
                            dpg.add_button(
                                label="Save PCA image",
                                width=-1,
                                callback=lambda s, d: self.save_pca_image(),
                            )

                    with dpg.child_window(
                        label="Postprocessing",
                        width=-1,
                        height=-1,
                        menubar=True,
                        no_scrollbar=True,
                    ):
                        with dpg.menu_bar():
                            with dpg.menu(label="Postprocessing", enabled=False):
                                pass
                        with dpg.group(horizontal=True):
                            dpg.add_text("Sharpen".rjust(LABEL_PAD))
                            dpg.add_checkbox(default_value=False)

                        with dpg.group(horizontal=True):
                            dpg.add_text("Remove noise".rjust(LABEL_PAD))
                            dpg.add_checkbox(default_value=False)

                        with dpg.group(horizontal=True):
                            dpg.add_text("Remove fisheye".rjust(LABEL_PAD))
                            dpg.add_checkbox(default_value=False)

                with dpg.child_window(
                    border=False, width=-1, height=-1, tag="images_wrapper"
                ):
                    with dpg.collapsing_header(
                        label="Project gallery",
                        tag="gallery_collapsible",
                        default_open=True,
                    ):
                        with dpg.child_window(border=True, tag="gallery_wrapper"):
                            with dpg.group(tag=self.gallery_tag):
                                dpg.add_text(
                                    tag="invalid_directory_message",
                                    default_value="No images found in specified directory",
                                )

                    with dpg.child_window(width=-1, border=True):
                        with dpg.group(tag="pca_wrapper", horizontal=True):
                            with dpg.plot(
                                show=True,
                                width=400,
                                tag="histogram_plot",
                                no_highlight=True,
                                anti_aliased=True,
                                no_box_select=True,
                                pan_mod=0,
                            ):
                                dpg.add_plot_axis(
                                    dpg.mvXAxis,
                                    tag="histogram_x",
                                )
                                dpg.add_plot_axis(
                                    dpg.mvYAxis,
                                    tag="histogram_y",
                                    no_tick_labels=True,
                                    no_tick_marks=True,
                                )
                                dpg.add_histogram_series(
                                    x=[],
                                    parent=dpg.last_item(),
                                    bins=255,
                                    density=True,
                                    max_range=255,
                                    outliers=True,
                                )
                            with dpg.plot(
                                show=True,
                                width=-1,
                                tag="spectrum_plot",
                                no_highlight=True,
                                anti_aliased=True,
                            ):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(
                                    dpg.mvXAxis,
                                    tag="spectrum_x",
                                )
                                dpg.add_plot_axis(
                                    dpg.mvYAxis,
                                    tag="spectrum_y",
                                    lock_min=True,
                                    lock_max=True,
                                )

                                ticks = (
                                    np.array(
                                        [
                                            np.linspace(400, 1000, 9),
                                            np.linspace(1, 204, 9),
                                        ]
                                    )
                                    .T.astype(int)
                                    .tolist()
                                )

                                labels = tuple([(str(l[0]), l[1]) for l in ticks])
                                dpg.set_axis_ticks("spectrum_x", labels)
                                dpg.set_axis_limits("spectrum_x", 1, 204)
                                dpg.configure_item(
                                    "spectrum_x", lock_max=True, lock_min=True
                                )

            with dpg.window(
                label="Settings",
                tag="settings_modal",
                show=False,
                no_move=True,
                no_collapse=True,
                modal=True,
                width=700,
                height=400,
                no_resize=True,
            ):
                ...

            with dpg.window(
                label="Results",
                tag="results_window",
                show=False,
                autosize=True,
                no_resize=True,
                width=512,
                height=512,
                no_focus_on_appearing=True,
            ):
                with dpg.tab_bar(tag="results_tabs"):
                    dpg.add_tab(label="Channels", tag="channel_image_tab")
                    dpg.add_tab(label="PCA", tag="pca_image_tab")

            w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
            with dpg.window(
                modal=True,
                no_background=True,
                no_move=True,
                no_scrollbar=True,
                no_title_bar=True,
                no_close=True,
                no_resize=True,
                tag="loading_indicator",
                show=False,
                pos=(w // 2 - 100, h // 2 - 100),
            ):
                dpg.add_loading_indicator(radius=20)
                dpg.add_button(
                    label="Loading hyperspectral data...",
                    indent=30,
                    tag="loading_indicator_message",
                )
                dpg.bind_item_theme(dpg.last_item(), self.button_theme)

            with dpg.file_dialog(
                callback=self.directory_picker_callback,
                modal=True,
                show=False,
                directory_selector=True,
                width=800,
                height=600,
                tag="project_directory_picker",
            ):
                ...

            with dpg.texture_registry():
                dpg.add_raw_texture(
                    512,
                    512,
                    np.ones((512, 512, 3)),  # type:ignore
                    tag="pca_images",
                    format=dpg.mvFormat_Float_rgb,
                )

            with dpg.texture_registry():
                dpg.add_raw_texture(
                    512,
                    512,
                    np.ones((512, 512, 3)),  # type:ignore
                    tag="channel_images",
                    format=dpg.mvFormat_Float_rgb,
                )

    def on_key_ctrl(self):
        if dpg.is_key_pressed(dpg.mvKey_Q):
            dpg.stop_dearpygui()
            dpg.destroy_context()

        if dpg.is_key_pressed(dpg.mvKey_S):
            self.save_all_shown_images()
        if dpg.is_key_pressed(dpg.mvKey_O):
            dpg.show_item("project_directory_picker")

        if dpg.is_key_pressed(dpg.mvKey_Comma):
            if not dpg.is_item_visible("settings_modal"):
                self.show_settings_modal()
            else:
                dpg.hide_item("settings_modal")

        if dpg.is_key_down(dpg.mvKey_Alt):
            if dpg.is_key_down(dpg.mvKey_Shift):
                if dpg.is_key_pressed(dpg.mvKey_M):
                    dpg.show_tool(dpg.mvTool_Metrics)
            elif dpg.is_key_pressed(dpg.mvKey_M):
                menubar_visible = dpg.get_item_configuration(self.window_tag)["menubar"]
                dpg.configure_item(self.window_tag, menubar=not menubar_visible)

    def close_results_window(self):
        if dpg.does_item_exist("results_window"):
            dpg.hide_item("results_window")

    def setup_handler_registries(self):
        with dpg.handler_registry():
            dpg.add_key_down_handler(dpg.mvKey_Control, callback=self.on_key_ctrl)
            dpg.add_key_down_handler(
                dpg.mvKey_Escape, callback=self.close_results_window
            )
            dpg.add_key_down_handler(dpg.mvKey_Escape, callback=self.hide_modals)
            dpg.add_key_press_handler(
                dpg.mvKey_F11, callback=lambda: dpg.toggle_viewport_fullscreen()
            )
            dpg.add_mouse_click_handler(
                dpg.mvMouseButton_Left, callback=self.handle_lmb
            )

        with dpg.item_handler_registry(tag="collapsible_clicked_handler"):
            dpg.add_item_clicked_handler(callback=self.collapsible_clicked_callback)

        with dpg.item_handler_registry(tag="window_resize_handler"):
            dpg.add_item_resize_handler(callback=self.window_resize_callback)

    def hide_modals(self):
        if dpg.is_item_visible("settings_modal"):
            dpg.hide_item("settings_modal")
        dpg.hide_item("project_directory_picker")

    def show_settings_modal(self):
        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
        dpg.configure_item("settings_modal", pos=[w // 2 - 350, h // 2 - 200])
        dpg.show_item("settings_modal")

    def collapsible_clicked_callback(self, _sender=None, _data=None):
        if dpg.does_item_exist("image_gallery"):
            dpg_gallery_visible = dpg.is_item_visible("image_gallery")
        else:
            return
        gallery_visible: bool = (
            dpg.get_item_state("gallery_collapsible")["clicked"] != dpg_gallery_visible
        )
        pca_visible = dpg.is_item_visible("pca_wrapper")
        vp_height = dpg.get_viewport_height()

        if gallery_visible and not pca_visible:
            dpg.configure_item("gallery_wrapper", height=-50)

        if gallery_visible and pca_visible:
            dpg.configure_item("gallery_wrapper", height=vp_height // 2)
            dpg.configure_item("pca_wrapper", height=-1)

        if not gallery_visible and pca_visible:
            dpg.configure_item("pca_wrapper", height=-1)

    def bind_item_handlers(self):
        dpg.bind_item_handler_registry(
            "gallery_collapsible", "collapsible_clicked_handler"
        )
        dpg.bind_item_handler_registry(self.window_tag, "window_resize_handler")

    @partial(loading_indicator, message="Loading hyperspectral data...")
    def image_select_callback(self, sender, _data):
        for i in self.project.images:
            dpg.bind_item_theme(f"image_{i}_button", self.normal_button_theme)
        dpg.bind_item_theme(sender, self.active_button_theme)
        dpg.show_item("histogram_plot")

        image_label: str | None = dpg.get_item_user_data(sender)
        if image_label is None:
            return
        _, img = self.project.set_current_image(image_label)
        img.load_hsi_image()
        assert isinstance(self.project.current_image, tuple)

        self.update_pca_images()
        self.update_channels_images()
        self.update_histogram_plot()

        dpg.delete_item("spectrum_y", children_only=True)
        dpg.delete_item("channel_image_plot", children_only=True, slot=0)

    @partial(loading_indicator, message="Reducing dimensions...")
    def update_pca_images(self):
        if not hasattr(self, "project"):
            return
        if self.project.current_image is None:
            return
        dpg.show_item("results_window")

        if not dpg.does_item_exist("pca_image_plot"):
            with dpg.plot(
                tag="pca_image_plot",
                parent="pca_image_tab",
                width=512,
                height=512,
                equal_aspects=True,
            ):
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    no_tick_labels=True,
                    no_tick_marks=True,
                    tag="pca_x_axis",
                )
                with dpg.plot_axis(
                    dpg.mvYAxis,
                    no_tick_labels=True,
                    no_tick_marks=True,
                    tag="pca_y_axis",
                ):
                    dpg.add_image_series(
                        "pca_images",
                        [0, 0],
                        [512, 512],
                        tag="pca_image",
                        show=False,
                        tint_color=(255, 255, 255, 100),
                    )

        _, image = self.project.current_image
        n_components = dpg.get_value("pca_n_components")
        reducer = dpg.get_value("image_reducer")
        if reducer == "PCA":
            image.reducer = PCA(n_components=n_components, svd_solver="auto")
        elif reducer == "ICA":
            image.reducer = FastICA(n_components=n_components)
        else:
            image.reducer = PCA(n_components=n_components, svd_solver="auto")

        image.reduce_hsi_dimensions(n_components)
        dpg.configure_item("pca_slider", max_value=dpg.get_value("pca_n_components"))

        apply_clahe = dpg.get_value("apply_clahe_checkbox")

        self.pca_images = image.pca_images(apply_clahe)
        self.rotate_images()
        self.pca_component_slider_callback(None, dpg.get_value("pca_slider"))
        dpg.show_item("pca_image")
        dpg.fit_axis_data("pca_y_axis")
        dpg.fit_axis_data("pca_x_axis")

    def pca_component_slider_callback(self, _sender, data):
        if not dpg.does_item_exist("pca_images"):
            return
        if not isinstance(self.pca_images, np.ndarray):
            return

        image_index = data - 1

        dpg.set_value("pca_images", self.pca_images[image_index])
        dpg.set_value("results_tabs", "pca_image_tab")

        if dpg.get_value("histogram_source_combo") == "PCA":
            self.update_histogram_plot()

    def channel_slider_callback(self, _sender, data):
        if not dpg.does_item_exist("channel_images"):
            return
        if self.project.current_image is None:
            return
        if not isinstance(self.project.current_image[1].hsi_image, np.ndarray):
            return
        if self.channel_images is None:
            return

        images = self.channel_images

        image_index = data - 1

        dpg.set_value("channel_images", images[image_index])
        dpg.set_value("results_tabs", "channel_image_tab")
        if dpg.get_value("histogram_source_combo") == "Channel":
            self.update_histogram_plot()

    @log_exec_time
    @partial(loading_indicator, message="Extracting channels...")
    def update_channels_images(self):
        if self.project.current_image is None:
            return
        dpg.show_item("results_window")

        if not dpg.does_item_exist("channel_image_plot"):
            with dpg.plot(
                parent="channel_image_tab",
                width=512,
                height=512,
                equal_aspects=True,
                tag="channel_image_plot",
            ):
                dpg.add_plot_axis(
                    dpg.mvXAxis,
                    no_tick_labels=True,
                    no_tick_marks=True,
                    tag="channel_x_axis",
                )
                with dpg.plot_axis(
                    dpg.mvYAxis,
                    no_tick_labels=True,
                    no_tick_marks=True,
                    tag="channel_y_axis",
                ):
                    dpg.add_image_series(
                        "channel_images",
                        [0, 0],
                        [512, 512],
                        tag="channel_image",
                        show=False,
                        tint_color=(255, 255, 255, 100),
                    )

        _, image = self.project.current_image
        if (images := image.channel_images()) is None:
            return

        self.channel_images = images
        self.rotate_images()
        self.channel_slider_callback(None, dpg.get_value("channel_slider"))
        dpg.show_item("channel_image")
        dpg.fit_axis_data("channel_y_axis")
        dpg.fit_axis_data("channel_x_axis")

    def window_resize_callback(self, _sender=None, _data=None):
        if not hasattr(self, "project"):
            return
        if not self.project.images:
            return

        available_width = dpg.get_item_rect_size("gallery_collapsible")[0]
        image_width = (available_width - 160) // self.image_gallery_n_columns

        for label in self.project.images.keys():
            dpg.set_item_width(f"image_{label}_wrapper", image_width)
            dpg.set_item_height(f"image_{label}_button", image_width)

        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
        if dpg.does_item_exist("loading_indicator"):
            dpg.configure_item("loading_indicator", pos=(w // 2 - 100, h // 2 - 100))

        self.collapsible_clicked_callback()

        dpg.set_item_width("spectrum_plot", -1)
        dpg.set_item_width("histogram_plot", available_width // 2)

    def update_histogram_plot(self):
        self.collapsible_clicked_callback()

        if self.project.current_image is None:
            return
        _, image = self.project.current_image
        axis_user = dpg.get_value("histogram_axis_combo")
        source_user = dpg.get_value("histogram_source_combo")

        if axis_user == "Rows":
            axis = 0
        else:
            axis = 1

        if not isinstance(self.pca_images, np.ndarray):
            current_pca_image = image.png_image
        else:
            current_pca_image = self.pca_images[dpg.get_value("pca_slider") - 1]

        if not isinstance(self.channel_images, np.ndarray):
            current_channel_image = image.png_image
        else:
            current_channel_image = self.channel_images[
                dpg.get_value("channel_slider") - 1
            ]

        if source_user == "HSI":
            source = image.hsi_image
            source_type = ImageType.HSI
        elif source_user == "PNG":
            source = image.png_image
            source_type = ImageType.PNG
        elif source_user == "PCA":
            source = current_pca_image
            source_type = ImageType.PCA
        else:
            source = current_channel_image
            source_type = ImageType.CHANNEL

        with dpg.mutex():
            dpg.delete_item("histogram_y", children_only=True)
            dpg.add_histogram_series(
                image.histogram_data(source, axis, source_type=source_type),
                parent="histogram_y",
                bins=dpg.get_value("histogram_bins"),
                density=True,
                max_range=255,
                min_range=0,
                outliers=True,
            )
            dpg.fit_axis_data("histogram_x")
            dpg.fit_axis_data("histogram_y")

    def rotate_images(self):
        if self.pca_images is None:
            return
        if self.channel_images is None:
            return
        if self.project.current_image is None:
            return

        angle = int(dpg.get_value("rotate_images"))
        _, image = self.project.current_image
        if (channel_images := image.channel_images()) is None:
            return
        apply_clahe = dpg.get_value("apply_clahe_checkbox")
        if (pca_images := image.pca_images(apply_clahe)) is None:
            return

        for i, img in enumerate(pca_images):
            self.pca_images[i] = rotate_image(img, angle)

        for i, img in enumerate(channel_images):
            self.channel_images[i] = rotate_image(img, angle)

        dpg.set_value("pca_images", self.pca_images[dpg.get_value("pca_slider") - 1])
        dpg.set_value(
            "channel_images", self.channel_images[dpg.get_value("channel_slider") - 1]
        )

    def drag_point_callback(self, sender):
        drag_data = dpg.get_value(sender)
        point = [512 - int(drag_data[1]), int(drag_data[0])]
        if point[0] >= 512:
            point[0] = 511
        if point[1] >= 512:
            point[1] = 511
        if point[0] < 0:
            point[0] = 0
        if point[1] < 0:
            point[1] = 0
        assert isinstance(self.channel_images, np.ndarray)
        try:
            spectrum = self.channel_images[:, *point, 0]
        except IndexError:
            return

        color = dpg.get_item_configuration(sender)["color"]
        plot_color = np.array(color) * [255, 255, 255, 255]
        alias = dpg.get_item_alias(sender)
        if not dpg.does_item_exist(f"spectrum_point_{alias}"):

            dpg.add_line_series(
                label=f"{point}",
                parent="spectrum_y",
                x=list(range(1, 205)),
                y=spectrum.tolist(),
                tag=f"spectrum_point_{alias}",
            )
            with dpg.theme() as plot_theme:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(
                        dpg.mvPlotCol_Line,
                        plot_color.tolist(),
                        category=dpg.mvThemeCat_Plots,
                    )
            dpg.bind_item_theme(f"spectrum_point_{alias}", plot_theme)
            dpg.fit_axis_data("spectrum_x")
        else:
            dpg.configure_item(
                f"spectrum_point_{alias}",
                y=spectrum.tolist(),
                label=f"{point}",
            )

    def add_drag_point(self, position):
        dp_id = uuid.uuid4().urn
        color = np.array(
            dpg.sample_colormap(dpg.mvPlotColormap_Jet, np.random.random())
        ) * [255, 255, 255, 255]
        dpg.add_drag_point(
            default_value=tuple(position),
            color=color.tolist(),
            callback=lambda s: self.drag_point_callback(s),
            tag=f"drag_point_{dp_id}",
            parent="channel_image_plot",
        )
        self.drag_point_callback(f"drag_point_{dp_id}")

    def handle_lmb(self):
        if not dpg.does_item_exist("channel_image_plot"):
            return

        plot_hovered = dpg.is_item_hovered("channel_image_plot")
        plot_focused = dpg.is_item_focused("channel_image_plot")
        ctrl_pressed = dpg.is_key_down(dpg.mvKey_Control)

        if plot_focused and plot_hovered and ctrl_pressed:
            plot_mouse_pos = dpg.get_plot_mouse_pos()
            self.add_drag_point(plot_mouse_pos)
