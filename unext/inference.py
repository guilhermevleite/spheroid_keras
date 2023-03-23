import argparse
import os
import cv2
from glob import glob
import yaml

import torch
import torch.backends.cudnn as cudnn

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import Resize

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
import archs

from settings import DATASETS_PATH, MODELS_PATH, OUTPUT_PATH
# from archs import UNext


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--dataset', default=None,
                        help='dataset name')
    parser.add_argument('--out', default=None,
                        help='output folder, only use if diferent than --name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open(MODELS_PATH + '/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    cudnn.benchmark = True

    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])
    model = model.cuda()

    print('DATASET', args.dataset)
    img_ids = glob(os.path.join(DATASETS_PATH,
                                args.dataset,
                                'images',
                                '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)
    # print('BEFORE', len(val_img_ids))
    val_img_ids = img_ids
    print('AFTER', len(val_img_ids))

    model.load_state_dict(torch.load(MODELS_PATH + '/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(DATASETS_PATH, args.dataset, 'images'),
        mask_dir=os.path.join(DATASETS_PATH, args.dataset, 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    # gput = AverageMeter()
    # cput = AverageMeter()

    if args.out != None:
        config['name'] = args.out

    for c in range(config['num_classes']):
        os.makedirs(os.path.join(OUTPUT_PATH,
                                 config['name'],
                                 str(c)), exist_ok=True)

        os.makedirs(os.path.join(OUTPUT_PATH,
                                 config['name'],
                                 str(c),
                                 'metric'), exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            # compute output
            output = model(input)

            iou, dice = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join(OUTPUT_PATH,
                                             config['name'],
                                             str(c),
                                             'metric',
                                             '{:.2f}_'.format(dice) + meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

                    cv2.imwrite(os.path.join(OUTPUT_PATH,
                                             config['name'],
                                             str(c),
                                             meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
