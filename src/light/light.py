import light_detection_module as det

# Pixel values above this value are ignored as 'too bright'
UPPER_LEVEL = 255

class Light:
    """getting the illumination"""

    def __init__(self, class_numbers):
        """init model"""
        UPPER_LEVEL = 255
        # TODO: Add more configs

    def get_light(self, frame) -> int:
        """getting the illumination of the frame"""
        return det.detect(frame, UPPER_LEVEL)
