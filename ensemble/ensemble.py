from pathlib import Path
import cv2 as cv
from torch import tensor
from torchmetrics.classification import Dice
from metrics import iou_score
import statistics
import numpy as np

PROTOCOL = 'protocol_5'
MODEL_ONE = 'TransUnet'
MODEL_TWO = 'SwinUnet'

INPUT_FOLDER = Path('/home/leite/Pictures', PROTOCOL)
OUTPUT_FOLDER = Path(f'/home/leite/Pictures/ensemble_{PROTOCOL}_{MODEL_ONE}_{MODEL_TWO}')
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

img_folder_one = Path(INPUT_FOLDER, MODEL_ONE)
img_folder_two = Path(INPUT_FOLDER, MODEL_TWO)

img_files = Path(img_folder_one).glob('*_img.png')
print(f'Found {img_files} images in {img_folder_one}')

dice_list = []
for f in img_files:
    file_name = f.name[-17:]
    gt_name = file_name.replace('_img', '_ori')
    inference_name = file_name.replace('_img', '')
    # print(file_name)


    path_one = Path(img_folder_one).glob(f'*{inference_name}')
    mask_one = cv.imread(str(list(path_one)[0]), 0)
    mask_one = mask_one.astype('float32') / 255

    path_two = Path(img_folder_two).glob(f'*{inference_name}')
    mask_two = cv.imread(str(list(path_two)[0]), 0)
    mask_two = mask_two.astype('float32') / 255

    if mask_one.shape[0] != mask_two.shape[0]:
        mask_two = cv.resize(mask_two, mask_one.shape)

    path_gt = Path(img_folder_one).glob(f'*{gt_name}')
    target = cv.imread(str(list(path_gt)[0]), 0)
    target = target.astype('float32') / 255
    tensor_target = tensor(target)

    ensemble = cv.bitwise_and(mask_one, mask_two)
    if target.shape[0] != ensemble.shape[0]:
        target = cv.resize(target, ensemble.shape)
    tensor_ensemble = tensor(ensemble)

    cv.imwrite(f'{str(OUTPUT_FOLDER)}/{inference_name}', np.multiply(ensemble, 255))
    cv.imwrite(f'{str(OUTPUT_FOLDER)}/{gt_name}', np.multiply(target, 255))

    iou, dice = iou_score(tensor_ensemble, tensor_target)
    dice_list.append(dice)
    # print(dice)
    # print()

print(f'{PROTOCOL} - {MODEL_ONE} - {MODEL_TWO}')
print(f'Average {statistics.mean(dice_list)}')
