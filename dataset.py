import os

import albumentations as A
import cv2
import natsort
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_MEAN = 0.449
IMAGE_STD = 0.226


class StenosisDataset(Dataset):
    def __init__(self, image_dir, mask_dir, is_train=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train

        self.image_filenames = natsort.natsorted([
            name for name in os.listdir(image_dir)
            if os.path.isfile(os.path.join(image_dir, name))
        ])
        mask_filenames = set([
            name for name in os.listdir(mask_dir)
            if os.path.isfile(os.path.join(mask_dir, name))
        ])

        if set(self.image_filenames) != mask_filenames:
            missing_masks = natsort.natsorted(set(self.image_filenames) - mask_filenames)
            missing_images = natsort.natsorted(mask_filenames - set(self.image_filenames))
            raise ValueError(
                "image/mask 파일명이 서로 맞지 않습니다. "
                f"missing_masks={missing_masks[:5]}, missing_images={missing_images[:5]}"
            )

        if is_train:
            # SSASS/MediPixel 표에 맞춘 stenosis 전용 augmentation입니다.
            # grayscale 영상이지만 HSV 계열 변환을 위해 augmentation 중에는 RGB로 바꿉니다.
            self.transform = A.Compose([
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.15, 0.15),
                    rotate=(-15, 15),
                    shear=(-3, 3),
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=1.0,
                ),
                A.Perspective(
                    scale=(0.0, 0.0005),
                    keep_size=True,
                    fit_output=False,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=0.3,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=3,
                    sat_shift_limit=30,
                    val_shift_limit=40,
                    p=0.5,
                ),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"마스크를 읽을 수 없습니다: {mask_path}")

        if self.transform is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            augmented = self.transform(image=image_rgb, mask=mask)
            image = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2GRAY)
            mask = augmented["mask"]

        image = image.astype(np.float32) / 255.0
        image = (image - IMAGE_MEAN) / IMAGE_STD
        mask = (mask > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        return image_tensor, mask_tensor, filename


def make_stenosis_loaders(data_dir, batch_size, num_workers):
    loaders = []
    for split in ["train", "val", "test"]:
        dataset = StenosisDataset(
            image_dir=os.path.join(data_dir, "stenosis", split, "images"),
            mask_dir=os.path.join(data_dir, "stenosis", split, "masks"),
            is_train=(split == "train"),
        )
        loaders.append(DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        ))
    return loaders


def denormalize_image(image_tensor):
    # 시각화용으로 정규화된 tensor를 다시 0~1 grayscale로 되돌립니다.
    image = image_tensor.detach().cpu().float().squeeze().numpy()
    image = image * IMAGE_STD + IMAGE_MEAN
    return np.clip(image, 0.0, 1.0)
