import random

from augly.image import transforms
import numpy as np
from PIL import Image
import os


def brightness(img, seed, **kwargs):
    np.random.seed(seed)
    factor = np.random.uniform(0.7, 1.3)
    img = transforms.Brightness(factor=factor, p=1.0)(img)  # 1 - 2
    return img


def blur(img, seed, **kwargs):
    np.random.seed(seed)
    radius = np.random.uniform(1, 3)
    img = transforms.Blur(radius=radius, p=1.0)(img)  # 1-3
    return img


def pixelization(img, seed, **kwargs):
    np.random.seed(seed)
    ratio = np.random.uniform(0.5, 1)
    img = transforms.Pixelization(ratio=ratio, p=1.0)(img)  # 0.5 - 1
    return img


def random_noise(img, seed, **kwargs):
    np.random.seed(seed)
    var = np.random.uniform(0.01, 0.03)
    img = transforms.RandomNoise(mean=0.0, var=var, p=1.0, seed=seed)(img)
    return img


def encoding_quality(img, seed, **kwargs):
    np.random.seed(seed)
    quality = np.random.randint(50, 80)
    img = transforms.EncodingQuality(quality=quality, p=1.0)(img)

    return img


def shuffle_pixels(img, seed, **kwargs):
    np.random.seed(seed)
    factor = np.random.uniform(0.05, 0.15)
    img = transforms.ShufflePixels(factor=factor, seed=seed, p=1.0)(img)

    return img


def sharpen(img, seed, **kwargs):
    np.random.seed(seed)
    factor = np.random.uniform(1, 5)
    img = transforms.Sharpen(factor=factor, p=1.0)(img)

    return img


def hflip(img, **kwargs):
    img = transforms.HFlip(p=1.0)(img)

    return img


def vflip(img, **kwargs):
    img = transforms.VFlip(p=1.0)(img)
    return img


def rotate(img, seed, **kwargs):

    np.random.seed(seed)
    if np.random.rand() < 0.5:
        img = transforms.Rotate(90)(img)
    else:
        img = transforms.Rotate(270)(img)
    return img


def grayscale(img, *args, **kwargs):
    img = transforms.Grayscale(p=1.0)(img)
    return img


def saturation(img, seed=None, **kwargs):
    np.random.seed(seed)
    factor = np.random.uniform(0.5, 1)
    img = transforms.Saturation(factor=factor, p=1.0)(img)
    return img


def scale(img, seed=None, **kwargs):
    np.random.seed(seed)
    factor = np.random.uniform(0.5, 1)
    img = transforms.Scale(factor=factor, p=1.0)(img)
    return img


def opacity(img, seed=None, **kwargs):
    np.random.seed(seed)
    level = np.random.uniform(0.5, 1)
    img = transforms.Opacity(level=level, p=1.0)(img)
    return img


def color_jitter(img, seed=None, **kwargs):
    np.random.seed(seed)
    axis = np.random.uniform(0.9, 1.1, size=3)
    x1, x2, x3 = axis
    img = transforms.ColorJitter(brightness_factor=x1, contrast_factor=x2, saturation_factor=x3)(img)
    return img


def pad(img, seed=None, **kwargs):
    np.random.seed(seed)
    axis = np.random.uniform(0.1, 0.3, size=2)
    x_pos, y_pos = axis[0], axis[1]

    np.random.seed(seed)
    rgb = np.random.randint(0, 255, size=3)
    r, g, b = rgb
    img = transforms.Pad(w_factor=x_pos, h_factor=y_pos, color=(r, g, b), p=1.0)(img)  # 0.1 - 0.3
    return img


def change_aspect_ratio(img, seed=None, **kwargs):
    np.random.seed(seed)
    ratio = np.random.uniform(0.5, 1.5)
    img = transforms.ChangeAspectRatio(ratio=ratio, p=1.0)(img)
    return img


def overlay_image(img, overlay_img=None, seed=None, **kwargs):
    opacity = np.random.uniform(0.3, 0.7)
    transforms.OverlayImage(overlay_img, opacity=opacity, x_pos=0., y_pos=0, p=1.0)(img)
    return img


def pad_and_emoji(img, seed=None, **kwargs):

    img = pad(img, seed)
    np.random.seed(seed)
    emoji_sizes = np.random.uniform(0.1, 0.2, size=4)
    s1, s2, s3, s4 = emoji_sizes
    np.random.seed(seed)
    opacities = np.random.uniform(0.5, 1, size=4)
    o1, o2, o3, o4 = opacities

    dir_name = os.path.dirname(transforms.OverlayEmoji.__init__.__defaults__[0])
    np.random.seed(seed)
    eps = [os.path.join(dir_name, x) for x in np.random.choice(os.listdir(dir_name), size=4)]

    axis1 = np.random.uniform(0, 0.2, size=4)
    axis2 = np.random.uniform(0.7, 0.9, size=4)
    img2 = transforms.OverlayEmoji(eps[0], opacity=o1, emoji_size=s1, x_pos=axis1[0], y_pos=axis1[1], p=1.0)(img)
    img3 = transforms.OverlayEmoji(eps[1], opacity=o2, emoji_size=s2, x_pos=axis2[0], y_pos=axis2[1], p=1.0)(img2)
    img4 = transforms.OverlayEmoji(eps[2], opacity=o3, emoji_size=s3, x_pos=axis1[2], y_pos=axis2[2], p=1.0)(img3)
    img = transforms.OverlayEmoji(eps[3], opacity=o4, emoji_size=s4, x_pos=axis2[3], y_pos=axis1[3], p=1.0)(img4)

    return img


def overlay_emoji(img, seed=None, **kwargs):
    dir_name = os.path.dirname(transforms.OverlayEmoji.__init__.__defaults__[0])
    np.random.seed(seed)
    emoji_path = os.path.join(dir_name, np.random.choice(os.listdir(dir_name)))
    np.random.seed(seed)
    opacity = np.random.uniform(0.5, 1)
    np.random.seed(seed)
    emoji_size = np.random.uniform(0.1, 0.4)
    np.random.seed(seed)
    axis = np.random.uniform(0, 0.8, size=2)
    x_pos, y_pos = axis[0], axis[1]
    img = transforms.OverlayEmoji(emoji_path=emoji_path, opacity=opacity, emoji_size=emoji_size,
                                  x_pos=x_pos, y_pos=y_pos, p=1.0)(img)
    return img


def overlay_text(img, seed=None, **kwargs):
    np.random.seed(seed)
    dir_name = os.path.dirname(transforms.OverlayText.__init__.__defaults__[1])
    fn = np.random.choice([x for x in os.listdir(dir_name) if x.endswith(".ttf")])
    font_file = os.path.join(dir_name, fn)
    np.random.seed(seed)
    opacity = np.random.uniform(0.5, 1)
    np.random.seed(seed)
    font_size = np.random.uniform(0.1, 0.2)
    np.random.seed(seed)
    axis = np.random.uniform(size=2)
    x_pos, y_pos = axis[0], axis[1]
    np.random.seed(seed)
    text = np.random.randint(200, size=5).tolist()

    np.random.seed(seed)
    rgb = np.random.randint(0, 255, size=3)
    r, g, b = rgb

    img = transforms.OverlayText(text=text, font_file=font_file, opacity=opacity, font_size=font_size,
                                 x_pos=x_pos, y_pos=y_pos, color=(r, g, b), p=1.0)(img)
    return img


def argument_img(img, seed, overlay_img=None):
    funcs = [
        [(random_noise, 1), (encoding_quality, 1), (shuffle_pixels, 1), (blur, 1), (pixelization, 1)],
        [(grayscale, 4), (brightness, 1), (saturation, 1), (opacity, 1), (sharpen, 1), (color_jitter, 1)],
        [(pad, 3), (scale, 1), (change_aspect_ratio, 1)],
        [(hflip, 1), (vflip, 1), (rotate, 1)],
        [(overlay_emoji, 2), (overlay_text, 1), (overlay_image, 2), (pad_and_emoji, 2)]
    ]

    np.random.seed(seed)
    params = dict(img=img, seed=seed, overlay_img=overlay_img)
    funcs = np.random.choice(funcs)
    np.random.seed(seed)
    sub_funcs = [x[0] for x in funcs]
    prob = [x[1] for x in funcs]
    sum_ = sum(prob)
    prob = [x / sum_ for x in prob]
    func = np.random.choice(sub_funcs, p=prob)
    img = func(**params)
    return img


def stack_imgs(imgs, seed, **kwargs):
    np.random.seed(seed)
    prob = np.random.rand()
    if prob < 0.3:
        imgs = [argument_img(x, seed=seed + i) for i, x in enumerate(imgs)]
    elif prob < 0.6:
        imgs = [pad(x, seed=seed + i) for i, x in enumerate(imgs)]

    arrays = [np.array(x.resize((224, 224))) for x in imgs]
    np.random.seed(seed)
    np.random.shuffle(arrays)

    if len(imgs) == 2:
        np.random.seed(seed)
        prob = np.random.rand()
        if prob < 0.5:
            img = Image.fromarray(np.concatenate(arrays, axis=0))
        else:
            img = Image.fromarray(np.concatenate(arrays, axis=1))
    elif len(imgs) == 4:
        array1 = np.concatenate([arrays[0], arrays[1]], axis=0)
        array2 = np.concatenate([arrays[2], arrays[3]], axis=0)
        img = Image.fromarray(np.concatenate([array1, array2], axis=1))
    else:
        raise NotImplementedError

    return img















