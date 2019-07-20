from PIL import ImageGrab
import numpy as np
import cv2
def grab_clipboard_img():
    pil = ImageGrab.grabclipboard()
    if pil is None:
        return None
    bgr = np.array(pil)
    rgb = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return rgb