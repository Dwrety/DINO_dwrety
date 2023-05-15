# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from . import transforms as T


def build_transforms(cfg, is_train=True):
    if is_train:
        if len(cfg.AUGMENT.MULT_MIN_SIZE_TRAIN)>0:
            min_size = cfg.AUGMENT.MULT_MIN_SIZE_TRAIN
        else:
            min_size = cfg.AUGMENT.MIN_SIZE_TRAIN
        max_size = cfg.AUGMENT.MAX_SIZE_TRAIN
        flip_horizontal_prob = cfg.AUGMENT.FLIP_PROB_TRAIN
        # flip_vertical_prob = cfg.AUGMENT.VERTICAL_FLIP_PROB_TRAIN
        brightness = cfg.AUGMENT.BRIGHTNESS
        contrast = cfg.AUGMENT.CONTRAST
        saturation = cfg.AUGMENT.SATURATION
        hue = cfg.AUGMENT.HUE

        crop_prob = cfg.AUGMENT.CROP_PROB
        min_ious = cfg.AUGMENT.CROP_MIN_IOUS
        min_crop_size = cfg.AUGMENT.CROP_MIN_SIZE

    else:
        min_size = cfg.AUGMENT.MIN_SIZE_TEST
        max_size = cfg.AUGMENT.MAX_SIZE_TEST
        flip_horizontal_prob = 0.0

    fix_res = cfg.AUGMENT.INPUT_FIX_RES
    if cfg.AUGMENT.INPUT_FORMAT is not '':
        input_format = cfg.AUGMENT.INPUT_FORMAT
    elif cfg.AUGMENT.INPUT_TO_BGR255:
        input_format = 'bgr255'
    normalize_transform = T.Normalize(
        mean=cfg.AUGMENT.PIXEL_MEAN, std=cfg.AUGMENT.PIXEL_STD, format=input_format
    )
 
    transform = T.Compose(
        [
            T.RandomResize([min_size], max_size=max_size),
            T.RandomHorizontalFlip(flip_horizontal_prob),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform
