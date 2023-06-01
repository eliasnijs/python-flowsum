import shutil

import imgui
import readfcs

from fst_utils import get_dir_path, get_file_paths


def win_parameters(window_dim, som_parameters, mst_parameters, hcc_parameters):
    imgui.set_next_window_position(0, 40 + 360)
    imgui.set_next_window_size(480, window_dim[1] - 40)
    imgui.begin("Parameters", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)

    # FlowSOM_SOMParameters
    imgui.dummy(1, 10)
    imgui.text("SOM Parameters")
    imgui.dummy(1, 1)
    imgui.text("n_epochs")
    _, som_parameters.n_epochs = imgui.input_int("##n_epochs", som_parameters.n_epochs)
    imgui.text("sigma")
    _, som_parameters.sigma = imgui.slider_float(
        "##sigma_slider", som_parameters.sigma, 0, 1
    )
    _, som_parameters.sigma = imgui.input_float("##sigma_input", som_parameters.sigma)
    imgui.text("alpha")
    _, som_parameters.alpha = imgui.slider_float(
        "##alpha_slider", som_parameters.alpha, 0, 1
    )
    _, som_parameters.alpha = imgui.input_float("##alpha_input", som_parameters.alpha)
    imgui.text("neighbourhood_function")
    _, som_parameters.neighbourhood_function = imgui.input_text(
        "##neighbourhood_function", som_parameters.neighbourhood_function, 256
    )
    imgui.text("activiation_distance")
    _, som_parameters.activiation_distance = imgui.input_text(
        "##activiation_distance", som_parameters.activiation_distance, 256
    )

    # FlowSOM_MSTParameters
    imgui.dummy(1, 10)
    imgui.text("MST Parameters")
    imgui.dummy(1, 1)
    imgui.text("distance_metric")
    _, mst_parameters.distance_metric = imgui.input_text(
        "##distance_metric", mst_parameters.distance_metric, 256
    )

    # FlowSOM_HCCParameters
    imgui.dummy(1, 10)
    imgui.text("HCC Parameters")
    imgui.dummy(1, 1)
    imgui.text("n_clusters")
    _, hcc_parameters.n_clusters = imgui.input_int(
        "##n_clusters", hcc_parameters.n_clusters
    )
    imgui.text("linkage_method")
    _, hcc_parameters.linkage_method = imgui.input_text(
        "##linkage_method", hcc_parameters.linkage_method, 256
    )
    imgui.text("n_bootstrap")
    _, hcc_parameters.n_bootstrap = imgui.input_int(
        "##n_bootstrap", hcc_parameters.n_bootstrap
    )

    imgui.end()


def win_actions(window_dim, flowsom, file_paths, image):
    imgui.set_next_window_position(0, 0)
    imgui.set_next_window_size(window_dim[0], 40)
    imgui.begin("Actions", flags=imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR)
    if imgui.button("Run"):
        if len(file_paths) > 0:
            print("Run button clicked!")
            data = readfcs.ReadFCS(file_paths[0]).data
            data = data.drop("label", axis=1)
            flowsom.fit(data, verbose=True)
            flowsom.report(save="./.flowsom/report/", verbose=True)
            image.i_plot = 0
        else:
            print("No file was selected")
    imgui.same_line()
    if imgui.button("Export"):
        if image.i_plot >= 0:
            shutil.copytree("./.flowsom/report", get_dir_path() + "/report")
    imgui.end()


def win_select_files(file_paths):
    imgui.set_next_window_position(0, 40)
    imgui.set_next_window_size(480, 360)
    imgui.begin("files", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)
    for file_path in file_paths:
        imgui.text(file_path)
    if imgui.button("Upload File"):
        new_file_paths = get_file_paths()
        if new_file_paths:
            file_paths.clear()
            file_paths.append(new_file_paths)
    imgui.end()


def win_controlls(window_dim, image_controls):
    imgui.set_next_window_position(window_dim[0] - 480, 40)
    imgui.set_next_window_size(480, window_dim[1] - 40)
    imgui.begin("Viewport", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE)

    options = [
        "Mst (with metaclusters)",
        "Mst (no metaclusters)",
        "Grid (with metaclusters)",
        "Grid (no metaclusters)",
        "Feature Maps",
    ]
    combo_label = options[image_controls.i_plot]
    if imgui.begin_combo("Plot", combo_label):
        for i, option in enumerate(options):
            _, selected = imgui.selectable(option, i == image_controls.i_plot)
            if selected and image_controls.i_plot >= 0:
                image_controls.i_plot = i
        imgui.end_combo()
    _, image_controls.zoom = imgui.slider_float("Zoom", image_controls.zoom, 0.1, 10.0)

    imgui.end()


def win_main(window_dim, texture_id, image_controls):
    imgui.set_next_window_position(480, 40)
    imgui.set_next_window_size(window_dim[0] - 480 * 2, window_dim[1] - 40)
    imgui.begin(
        "Main",
        flags=(
            imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
            | imgui.WINDOW_NO_TITLE_BAR
            | imgui.WINDOW_ALWAYS_HORIZONTAL_SCROLLBAR
            | imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR
        ),
    )
    width, height = imgui.get_content_region_available()
    length = max(width * image_controls.zoom, height * image_controls.zoom)
    imgui.image(texture_id, length, length)
    imgui.end()
