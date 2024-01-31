from matplotlib.style import available
from numpy import imag

from utils import *

TOOLTIP_DELAY_SEC = 0.1
LABEL_PAD = 23


@dataclass
class UI:
    window_tag: Literal["primary_window"] = field(init=False, default="primary_window")
    project: Project = field(init=False)
    image_labels: list[str] = field(init=False, default_factory=list)
    gallery_tag: Literal["image_gallery"] = field(init=False, default="image_gallery")
    current_image: str = field(init=False)
    image_gallery_n_columns: Literal[9] = 9
    sidebar_width: Literal[350] = 350

    def __post_init__(self):
        dpg.create_context()
        dpg.create_viewport(title="hsistat", width=1920, height=1080, vsync=True)
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
        dpg.destroy_context()
        dpg.stop_dearpygui()

    def setup_themes(self):
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_ButtonTextAlign, 0.5, category=dpg.mvThemeCat_Core
                )

        with dpg.theme() as self.button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button, (37, 37, 38), category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive,
                    (37, 37, 38),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    (37, 37, 38),
                    category=dpg.mvThemeCat_Core,
                )

    def bind_themes(self):
        dpg.bind_theme(self.global_theme)

    def populate_image_gallery(self):
        if not self.project.image_labels:
            dpg.add_text(
                parent=self.gallery_tag,
                default_value="No images found in specified directory",
            )
        for i in range(
            len(self.project.image_labels) // self.image_gallery_n_columns + 1
        ):
            dpg.add_group(
                tag=f"image_gallery_row_{i}", parent=self.gallery_tag, horizontal=True
            )

        for i, label in enumerate(self.project.image_labels):
            width, height, channels, data = dpg.load_image(
                f"{self.project.directory}/{label}/{label}.png"
            )

            with dpg.texture_registry():
                dpg.add_static_texture(width, height, data, tag=f"image_{label}")

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
                    callback=self.image_select_callback,
                )
                dpg.add_button(label=label, indent=2)
                dpg.bind_item_theme(dpg.last_item(), self.button_theme)

    def setup_dev(self):
        self.project = Project(dpg.get_value("project_directory"))
        self.image_labels = self.project.image_labels
        self.populate_image_gallery()

    def setup_layout(self):
        with dpg.window(
            label="hsistat",
            tag=self.window_tag,
            horizontal_scrollbar=False,
            no_scrollbar=True,
            min_size=[160, 90],
        ):
            with dpg.menu_bar(tag="menu_bar"):
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Open new image", shortcut="(Ctrl+O)")
                    dpg.add_menu_item(
                        label="Open project directory", shortcut="(Ctrl+Shift+O)"
                    )
                    dpg.add_menu_item(
                        label="Open latest image", shortcut="(Ctrl+Shift+I)"
                    )
                    dpg.add_menu_item(label="Save", shortcut="(Ctrl+S)")
                    dpg.add_menu_item(label="Save As", shortcut="(Ctrl+Shift+S)")
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
                        shortcut="(Win+F)",
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
                    border=False, width=self.sidebar_width, tag="sidebar"
                ):
                    dpg.add_progress_bar(tag="table_progress", width=-1, height=19)
                    with dpg.tooltip("table_progress", delay=TOOLTIP_DELAY_SEC):
                        dpg.add_text("Current operation progress")

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
                                dpg.add_text("Image directory")
                                dpg.add_input_text(
                                    tag="project_directory",
                                    default_value="/run/media/puglet5/HP P600/IH/stripes",
                                    width=120,
                                )
                                dpg.add_button(label="Browse", width=-1)

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
                                dpg.add_text("Show '< LOD'".rjust(LABEL_PAD))
                                dpg.add_checkbox()

                    with dpg.child_window(
                        label="PCA",
                        width=-1,
                        height=-1,
                        menubar=True,
                        no_scrollbar=True,
                    ):
                        with dpg.menu_bar():
                            with dpg.menu(label="PCA", enabled=False):
                                pass
                        with dpg.group(horizontal=True):
                            dpg.add_text("Method".rjust(LABEL_PAD))
                            dpg.add_combo(
                                items=["auto", "full", "randomized"],
                                default_value="auto",
                                width=-1,
                            )

                with dpg.child_window(
                    border=False, width=-1, height=-1, tag="images_wrapper"
                ):
                    with dpg.collapsing_header(
                        label="Project gallery",
                        tag="gallery_collapsible",
                        default_open=True,
                    ):
                        with dpg.child_window(border=True, tag="gallery_wrapper"):
                            dpg.add_group(tag=self.gallery_tag)

                    with dpg.child_window(width=-1, border=True):
                        with dpg.group(tag="pca_wrapper"):
                            with dpg.plot(width=-1):
                                dpg.add_plot_axis(dpg.mvXAxis)
                                dpg.add_histogram_series(
                                    x=[1, 2],
                                    label="test",
                                    parent=dpg.last_item(),
                                    bins=255,
                                    density=True,
                                    max_range=255,
                                    outliers=True,
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

    def on_key_ctrl(self):
        if dpg.is_key_pressed(dpg.mvKey_Q):
            dpg.stop_dearpygui()
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
                dpg.configure_item(self.window_tag, menubar=(not menubar_visible))

    def setup_handler_registries(self):
        with dpg.handler_registry():
            dpg.add_key_down_handler(dpg.mvKey_Control, callback=self.on_key_ctrl)
            dpg.add_key_down_handler(dpg.mvKey_Escape, callback=self.hide_modals)

        with dpg.item_handler_registry(tag="collapsible_clicked_handler"):
            dpg.add_item_clicked_handler(callback=self.collapsible_clicked_callback)

        with dpg.item_handler_registry(tag="window_resize_handler"):
            dpg.add_item_resize_handler(callback=self.window_resize_callback)

    def hide_modals(self):
        if dpg.is_item_visible("settings_modal"):
            dpg.hide_item("settings_modal")

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

    def image_select_callback(self, sender, data):
        print(sender, data)

    def window_resize_callback(self, _sender, _data):
        if not self.image_labels:
            return

        available_width = dpg.get_item_rect_size("gallery_collapsible")[0] - 160

        image_width = available_width // self.image_gallery_n_columns

        for label in self.image_labels:
            dpg.set_item_width(f"image_{label}_wrapper", image_width)
            dpg.set_item_height(f"image_{label}_button", image_width)

        self.collapsible_clicked_callback()
