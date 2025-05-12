from src.app import App
from src.light import Light
from src.pitscndcracks import PitsAndCracks
from src.segment import Segment

import os


app = App(
    models={
        "lightness": Light(),
        "markup quality": Segment(
            class_model=os.path.abspath("weights/mobilenet_class.pth"),
            segment_model=os.path.abspath("weights/r101best.pth")
        ),
        "potholes": PitsAndCracks(
            model=os.path.abspath("weights/pits.pt")
        )
    }
)
