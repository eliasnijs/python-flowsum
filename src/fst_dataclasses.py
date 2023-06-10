from dataclasses import dataclass
from typing import Optional


@dataclass
class Fst_Image:
    width: int = 0
    height: int = 0
    image: Optional[str] = None


@dataclass
class Fst_ImageControls:
    redraw: bool = False
    i_plot: int = -1
    zoom: float = 1.0
    image: Fst_Image = Fst_Image()
