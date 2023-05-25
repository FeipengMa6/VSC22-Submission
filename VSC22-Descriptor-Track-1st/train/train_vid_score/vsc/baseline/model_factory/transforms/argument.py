from albumentations import ImageOnlyTransform
from augly.image.transforms import OverlayImage
import numpy as np
import os
from PIL import Image
import pickle
from augly.image import transforms
from augly.utils import FONTS_DIR
from augly.utils import EMOJI_DIR
import random
import lmdb
import io
import albumentations as A


def overlay_emoji(img, seed=None, **kwargs):
    dir_name = os.path.join(EMOJI_DIR, np.random.choice(os.listdir(EMOJI_DIR)))
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
    return np.array(img)


def overlay_text(img, seed=None, **kwargs):
    np.random.seed(seed)
    dir_name = FONTS_DIR
    ttf_fn = np.random.choice([x for x in os.listdir(dir_name) if x.endswith(".ttf")])

    prefix, ext = os.path.splitext(ttf_fn)
    pkl_fn = f"{prefix}.pkl"

    font_file = os.path.join(dir_name, ttf_fn)
    pkl_file = os.path.join(dir_name, pkl_fn)

    np.random.seed(seed)
    opacity = np.random.uniform(0.1, 1)
    np.random.seed(seed)
    font_size = np.random.uniform(0.1, 0.2)
    np.random.seed(seed)
    axis = np.random.uniform(0, 0.5, size=2)
    x_pos, y_pos = axis[0], axis[1]
    np.random.seed(seed)

    with open(pkl_file, "rb") as f:
        charset = np.array(pickle.load(f), dtype=np.int64)

    text_length = np.random.randint(10, 20)
    text = np.random.choice(charset, size=text_length).tolist()
    np.random.seed(seed)
    rgb = np.random.randint(0, 255, size=3)
    r, g, b = rgb
    try:
        img = transforms.OverlayText(text=text, font_file=font_file, opacity=opacity, font_size=font_size,
                                     x_pos=x_pos, y_pos=y_pos, color=(r, g, b), p=1.0)(img)
    except Exception as e:
        pass
    return np.array(img)


def square_emoji(img, seed=None, **kwargs):
    np.random.seed(seed)

    emoji_sizes = np.random.uniform(0.1, 0.2, size=4)
    s1, s2, s3, s4 = emoji_sizes
    np.random.seed(seed)
    opacities = np.random.uniform(0.5, 1, size=4)
    o1, o2, o3, o4 = opacities

    dir_name = os.path.join(EMOJI_DIR, np.random.choice(os.listdir(EMOJI_DIR)))
    np.random.seed(seed)
    eps = [os.path.join(dir_name, x) for x in np.random.choice(os.listdir(dir_name), size=4)]

    axis1 = np.random.uniform(0, 0.2, size=4)
    axis2 = np.random.uniform(0.7, 0.9, size=4)
    img2 = transforms.OverlayEmoji(eps[0], opacity=o1, emoji_size=s1, x_pos=axis1[0], y_pos=axis1[1], p=1.0)(img)
    img3 = transforms.OverlayEmoji(eps[1], opacity=o2, emoji_size=s2, x_pos=axis2[0], y_pos=axis2[1], p=1.0)(img2)
    img4 = transforms.OverlayEmoji(eps[2], opacity=o3, emoji_size=s3, x_pos=axis1[2], y_pos=axis2[2], p=1.0)(img3)
    img = transforms.OverlayEmoji(eps[3], opacity=o4, emoji_size=s4, x_pos=axis2[3], y_pos=axis1[3], p=1.0)(img4)

    return np.array(img)


def change_aspect_ratio(img, seed=None, **kwargs):
    np.random.seed(seed)
    ratio = np.random.uniform(0.5, 1.5)
    img = transforms.ChangeAspectRatio(ratio=ratio, p=1.0)(img)
    return np.array(img)


def opacity(img, seed=None, **kwargs):
    np.random.seed(seed)
    level = np.random.uniform(0.5, 1)
    img = transforms.Opacity(level=level, p=1.0)(img)
    return np.array(img)


def pad(img, seed=None, **kwargs):
    np.random.seed(seed)
    axis = np.random.uniform(0.0, 0.3, size=2)
    x_pos, y_pos = axis[0], axis[1]

    np.random.seed(seed)
    rgb = np.random.randint(0, 255, size=3)
    r, g, b = rgb
    img = transforms.Pad(w_factor=x_pos, h_factor=y_pos, color=(r, g, b), p=1.0)(img)  # 0.1 - 0.3
    return np.array(img)


class BaseTransform(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(BaseTransform, self).__init__(always_apply, p)

    def apply(self, image, **params):
        image = self._array2img(image)
        return np.array(self._transform(image))

    def get_transform_init_args_names(self):
        return ()

    @staticmethod
    def _array2img(image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return image

    def _transform(self, image):
        return


class OverlayEmoji(BaseTransform):
    def _transform(self, image):
        return overlay_emoji(image)


class SquareEmoji(BaseTransform):

    def _transform(self, image):
        return square_emoji(image)


class OverlayText(BaseTransform):
    def _transform(self, image):
        return overlay_text(image)


class AspectRatio(BaseTransform):
    def _transform(self, image):
        return change_aspect_ratio(image)


class Opacity(BaseTransform):
    def _transform(self, image):
        return opacity(image)


class CropAndPad(BaseTransform):
    def _transform(self, image):
        return pad(image)


class RandomOverlayCorners(BaseTransform):

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self._c1 = Image.open("../data/meta/bg_img/1_bg.png")
        self._c2 = Image.open("../data/meta/bg_img/2_bg.png")
        self._c3 = Image.open("../data/meta/bg_img/3_bg.png")
        self._c4 = Image.open("../data/meta/bg_img/4_bg.png")

    def _transform(self, image):
        img = image
        ratio = np.random.uniform(0.15, 0.2)
        short_size = int(min(img.size) * ratio)

        img.paste(self._c2.resize([short_size, short_size]), (0, 0))
        img.paste(self._c3.resize([short_size, short_size]), (img.size[0] - short_size, 0))
        img.paste(self._c1.resize([short_size, short_size]), (0, img.size[1] - short_size))
        img.paste(self._c4.resize([short_size, short_size]), (img.size[0] - short_size, img.size[1] - short_size))

        return np.array(img)


class RandomStackImages(BaseTransform):

    def __init__(self, lmdb_path, lmdb_size, width, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.lmdb_path = lmdb_path
        self.lmdb_size = lmdb_size
        self.width = width
        self.pipeline = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.RandomResizedCrop(self.width, self.width, scale=(0.5, 1), p=1),
            A.OneOf([
                A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=1),
                A.RandomBrightness((-0.2, 0.1), p=1),
                A.ToGray(p=1),
                A.HueSaturationValue(p=1)
            ], p=0.1),
            A.GaussNoise(p=0.1),
            A.GaussianBlur(p=0.1),
            A.RandomScale(p=0.1),
            A.Perspective(p=0.1),
            A.OneOf([
                CropAndPad(p=1),
                A.CropAndPad(percent=(-0.4, 0.4), p=1)
            ], p=0.1),
        ])  # simple argument

    def _transform(self, image):
        if not hasattr(self, "lmdb_env") or self.lmdb_env is None:
            self._open_lmdb()

        img = image

        stack_img_nums = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])

        sampled_ids = np.random.randint(0, self.length, stack_img_nums)
        with self.lmdb_env.begin() as doc:
            other_imgs = [Image.open(io.BytesIO(doc.get(str(x).encode()))) for x in sampled_ids]

        imgs = [Image.fromarray(self.pipeline(image=np.array(x))["image"]) for x in other_imgs]

        imgs.append(img)

        random.shuffle(imgs)
        arrays = [np.array(x.resize([self.width, self.width])) for x in imgs]

        if len(arrays) == 2:
            prob = np.random.rand()
            if prob < 0.5:
                img = Image.fromarray(np.concatenate(arrays, axis=0))
            else:
                img = Image.fromarray(np.concatenate(arrays, axis=1))
        elif len(arrays) == 3:
            prob = np.random.rand()
            if prob < 0.5:
                img = Image.fromarray(np.concatenate(arrays, axis=0))
            else:
                img = Image.fromarray(np.concatenate(arrays, axis=1))
        elif len(arrays) == 4:
            array1 = np.concatenate([arrays[0], arrays[1]], axis=0)
            array2 = np.concatenate([arrays[2], arrays[3]], axis=0)
            img = Image.fromarray(np.concatenate([array1, array2], axis=1))
        else:
            raise NotImplementedError

        return np.array(img)

    def _open_lmdb(self):
        self.lmdb_env = lmdb.open(
            self.lmdb_path,
            map_size=int(self.lmdb_size),
            readonly=True,
            readahead=False,
            max_readers=8192,
            max_spare_txns=8192,
            lock=False
        )
        self.length = self.lmdb_env .stat()['entries']


class RandomOverlayImages(RandomStackImages):

    def _transform(self, image):
        if not hasattr(self, "lmdb_env") or self.lmdb_env is None:
            self._open_lmdb()

        img = image

        sampled_id = np.random.randint(0, self.length)
        with self.lmdb_env.begin() as doc:
            overlay_img = Image.open(io.BytesIO(doc.get(str(sampled_id).encode())))
        overlay_img = Image.fromarray(self.pipeline(image=np.array(overlay_img))["image"])

        opacity = np.random.uniform(0.2, 0.8)
        overlay_size = np.random.uniform(0.5, 1)
        img = OverlayImage(overlay_img, opacity=opacity, overlay_size=overlay_size, x_pos=0., y_pos=0, p=1.0)(img)

        return np.array(img)


class RandomCompose(A.Compose):

    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1.0,
                 shuffle=True):
        super().__init__(transforms, bbox_params, keypoint_params, additional_targets, p)
        self.shuffle = shuffle

    def __call__(self, *args, force_apply=False, **data):
        if self.shuffle:
            random.shuffle(self.transforms)
        return super().__call__(*args, force_apply=False, **data)









