from tkinter import Tk, filedialog

import pygame

from fst_dataclasses import Fst_Image


def get_file_paths():
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilename()
    return file_paths


def get_dir_path():
    root = Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory(mustexist=False)
    return directory_path


def image_load(path):
    image = pygame.image.load(path)
    image = pygame.transform.flip(image, False, True)
    image_str = pygame.image.tostring(image, "RGBA", 1)
    return Fst_Image(image.get_rect().width, image.get_rect().height, image_str)
