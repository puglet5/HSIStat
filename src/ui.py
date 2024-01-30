from utils import *

TOOLTIP_DELAY_SEC = 0.1
LABEL_PAD = 23


@dataclass
class UI:
    window_tag: str = field(init=False, default="primary_window")
    project: Project = field(init=False)
    image_gallery_n_columns = 9

    def __post_init__(self):
        dpg.create_context()
        dpg.create_viewport(title="hsistat", width=1920, height=1080, vsync=True)
        dpg.configure_app(wait_for_input=False)
        self.setup_handler_registries()
        self.setup_layout()

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

    def populate_image_gallery(self):
        for i in range(
            len(self.project.image_labels) // self.image_gallery_n_columns + 1
        ):
            print(i)
            dpg.add_group(
                tag=f"image_gallery_row_{i}", parent="image_gallery", horizontal=True
            )

        for i, label in enumerate(self.project.image_labels):
            width, height, channels, data = dpg.load_image(
                f"{self.project.folder}/{label}/{label}.png"
            )

            print(i // self.image_gallery_n_columns)

            with dpg.texture_registry():
                dpg.add_static_texture(width, height, data, tag=f"image_{label}")

            with dpg.group(
                parent=f"image_gallery_row_{i//self.image_gallery_n_columns}",
            ):
                dpg.add_image_button(
                    f"image_{label}",
                    width=150,
                    height=150,
                )
                dpg.add_text(label)

    def setup_dev(self):
        self.project = Project(dpg.get_value("project_folder"))
        self.populate_image_gallery()

    def setup_layout(self):
        with dpg.window(
            label="hsistat",
            tag=self.window_tag,
            horizontal_scrollbar=False,
            no_scrollbar=True,
        ):
            with dpg.menu_bar(tag="menu_bar"):
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Open new image", shortcut="(Ctrl+O)")
                    dpg.add_menu_item(
                        label="Open project folder", shortcut="(Ctrl+Shift+O)"
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
                with dpg.child_window(border=False, width=350, tag="sidebar"):
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
                                dpg.add_text("Image folder")
                                dpg.add_input_text(
                                    tag="project_folder",
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
                    with dpg.collapsing_header(label="Project gallery"):
                        with dpg.child_window(
                            border=True,
                            tag="image_gallery",
                        ):
                            ...

                    with dpg.child_window(width=-1, border=True):
                        ...

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

    def hide_modals(self):
        if dpg.is_item_visible("settings_modal"):
            dpg.hide_item("settings_modal")

    def show_settings_modal(self):
        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
        dpg.configure_item("settings_modal", pos=[w // 2 - 350, h // 2 - 200])
        dpg.show_item("settings_modal")
