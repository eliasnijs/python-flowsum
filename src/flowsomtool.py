import ctypes

import imgui
import imgui.integrations.sdl2
import sdl2
from OpenGL import GL as gl

from flowsom import FlowSOM
from fst_dataclasses import Fst_ImageControls
from fst_utils import image_load
from fst_windows import (
    win_actions,
    win_controlls,
    win_main,
    win_parameters,
    win_select_files,
)


def main():
    # Setup Gui context
    sdl2.SDL_Init(sdl2.SDL_INIT_EVERYTHING)
    window = sdl2.SDL_CreateWindow(
        b"FlowSOMTool",
        sdl2.SDL_WINDOWPOS_UNDEFINED,
        sdl2.SDL_WINDOWPOS_UNDEFINED,
        1920,
        960,
        sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_RESIZABLE,
    )
    gl_context = sdl2.SDL_GL_CreateContext(window)
    imgui.create_context()
    imgui.core.style_colors_light()
    renderer = imgui.integrations.sdl2.SDL2Renderer(window)

    # Setup flowsom context
    file_paths = []
    flowsom = FlowSOM()
    image = Fst_ImageControls()

    # Start program loop
    running = True
    while running:
        # Handle input
        event = sdl2.SDL_Event()
        while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == sdl2.SDL_QUIT:
                running = False
            renderer.process_event(event)
        renderer.process_inputs()

        # Get window context
        width = ctypes.c_int(0)
        height = ctypes.c_int(0)
        window_dim = sdl2.SDL_GetWindowSize(window, width, height)
        window_dim = (width.value, height.value)

        # Setup plot
        texture_id = gl.glGenTextures(1)
        prev_plot = image.i_plot

        # Configure imgui windows
        imgui.new_frame()
        win_actions(window_dim, flowsom, file_paths, image)
        win_parameters(
            window_dim, flowsom.som_param, flowsom.mst_param, flowsom.hcc_param
        )
        win_select_files(file_paths)
        win_controlls(window_dim, image)

        if (image.i_plot >= 0 and image.i_plot != prev_plot) or image.redraw:
            image.redraw = False
            plotnames = [
                "mst_wclusters.png",
                "mst_noclusters.png",
                "som_noclusters.png",
                "som_wclusters.png",
                "feature_planes.png",
            ]
            image.image = image_load(
                ".flowsom/report/images/" + plotnames[image.i_plot]
            )

        win_main(window_dim, texture_id, image)

        # Render everything
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        if image.i_plot >= 0:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGBA,
                image.image.width,
                image.image.height,
                0,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                image.image.image,
            )

        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        imgui.render()
        renderer.render(imgui.get_draw_data())
        sdl2.SDL_GL_SwapWindow(window)

        # Cleanup resources
        gl.glDeleteTextures(texture_id)

    # Cleanup resources
    sdl2.SDL_GL_DeleteContext(gl_context)
    sdl2.SDL_DestroyWindow(window)
    sdl2.SDL_Quit()


if __name__ == "__main__":
    main()
