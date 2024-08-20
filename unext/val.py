import argparse
# TODO replace os with pathlib
import os
from glob import glob

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
# from albumentations import Resize
# import time
# from archs import UNext

# from .settings import DATASETS_PATH, MODELS_PATH


# DATASETS_PATH = '/workspace/deep_learning/datasets/segmentation'
DATASETS_PATH = '/media/DATASETS'
# MODELS_PATH = '/workspace/models_free_space/w30/val_delete_later'
MODELS_PATH = '/media/MODELS'
OUTPUTS_PATH = '/media/INFERENCES'


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    parser.add_argument('--device', default='cpu', type=str,
                        help='Select GPU device (eg. cuda:0), default: cpu')
    parser.add_argument('--dataset', default=None,
                        help='Dataset name for validation, default: same dataset as training')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    with open(MODELS_PATH+'/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        if args.dataset != None:
            config['dataset'] = args.dataset

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['arch'])
    # model = archs.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])

    model = archs.__dict__[config['arch']](num_classes=config['num_classes'],
                                           input_channels=config['input_channels'],
                                           deep_supervision=config['deep_supervision'])

    model = model.to(args.device)

    # Data loading code
    img_ids = glob(os.path.join(DATASETS_PATH,
                                config['dataset'],
                                'images',
                                '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.97)

    print(f'{len(val_img_ids)} images found')
    model_path = f'{MODELS_PATH}/{args.name}/model.pth'
    print(model_path)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))
    model.eval()

    # val_transform = Compose([
    #     Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])
    val_transform = None

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(DATASETS_PATH, config['dataset'], 'images'),
        mask_dir=os.path.join(DATASETS_PATH, config['dataset'], 'masks'),
        img_size=config['input_h'], # TODO Delete this after fixing transforms
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

    # count = 0
    for c in range(config['num_classes']):
        os.makedirs(os.path.join(OUTPUTS_PATH, config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            # input = input.cuda()
            # target = target.cuda()
            # model = model.cuda()
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
                    cv2.imwrite(os.path.join(OUTPUTS_PATH, config['name'], str(c), meta['img_id'][i] + '.jpg'),
                                (output[i, c] * 255).astype('uint8'))

    print('IoU: %.4f' % iou_avg_meter.avg)
    print('Dice: %.4f' % dice_avg_meter.avg)

    # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
