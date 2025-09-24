from PIL import Image
import numpy as np

def resize_to_square(img: Image.Image, size: int):
    return img.resize((size, size), Image.Resampling.LANCZOS)

def center_crop(img: Image.Image, target_size: int):
    w, h = img.size
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))
    return img.resize((target_size, target_size), Image.Resampling.LANCZOS)

def pad_lr_to_720x480(img: Image.Image):
    # 把480x480图像pad成720x480
    img = img.resize((480, 480), Image.Resampling.LANCZOS)
    new_img = Image.new("RGB", (720, 480), (0, 0, 0))
    new_img.paste(img, (120, 0))
    return new_img
