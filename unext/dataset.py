from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_ids,
                 img_dir,
                 mask_dir,
                 img_size,  # TODO Delete after fixing transforms
                 img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of
                albumentations. Defaults to None.

        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size  # TODO Delete after fixing transforms
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        im_path = Path(self.img_dir, f'{img_id}{self.img_ext}')
        img = cv.imread(str(im_path))
        if img is None:
            raise Exception(f"Image not loaded {str(im_path)}")
            exit()

        # TODO remove this and resize using augmentation pipeline
        img = cv.resize(img, (self.img_size, self.img_size))

        mask = []
        for i in range(self.num_classes):
            msk_path = Path(self.mask_dir, str(i), img_id, self.mask_ext)
            msk_path = Path(self.mask_dir, str(i), f'{img_id}{self.mask_ext}')
            m = cv.imread(str(msk_path), cv.IMREAD_GRAYSCALE)
            if m is None:
                raise Exception(f"Image not loaded {str(msk_path)}")
                exit()

            # TODO remove this and resize using augmentation pipeline
            m = cv.resize(m, (self.img_size, self.img_size))

            mask.append(m[..., None])
        mask = np.dstack(mask)

        # TODO : apply transforms here
        if self.transform is not None:
            # augmented = self.transform(image=img, mask=mask)
            # img = augmented['image']
            # mask = augmented['mask']
            img = self.transform(img)
            mask = self.transform(mask)

        # TODO : move this to transform group
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}
