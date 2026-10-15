import os

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DeterministicCLAHE(A.ImageOnlyTransform):


    def __init__(self, clip_limit: float = 2.0, tile_grid_size=(8, 8), p: float = 1.0):
        super().__init__(p=p)
        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tuple(int(t) for t in tile_grid_size)

    def apply(self, img, **params):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=self.tile_grid_size)
        l_ch = clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2RGB)


def get_projection_transforms() -> A.Compose:

    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    ])


def get_train_transforms() -> A.Compose:

    return A.Compose([
        A.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.2,
            p=0.8,
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.7),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(8, 20), hole_width_range=(8, 40), fill=0, p=0.3),
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=0, p=0.5),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transforms() -> A.Compose:

    return A.Compose([
        DeterministicCLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])