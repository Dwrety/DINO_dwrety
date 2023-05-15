# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np

import torch
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder, clean_state_dict
import util.misc as utils
import data
from data import build_dataset, make_data_loader, try_to_find
from engine import do_inference, MetricLogger, DetectronCheckpointer


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--weight', '-w', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/comp_robot/cv_public_dataset/COCO2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='OUTPUT',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def build_model_main(cfg):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert cfg.MODEL.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(cfg.MODEL.modelname)
    # model, criterion, postprocessors = build_func(args)
    # return model, criterion, postprocessors
    return build_func(cfg)


def main(args):
    
    utils.init_distributed_mode(args)

    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # # setup logger
    args.output_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(
        output=os.path.join(args.output_dir, 'info.txt'), 
        distributed_rank=args.rank, 
        color=True, 
        name="groundingdino_inference")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))

    device = torch.device(args.device)

    # build model
    model = build_model_main(cfg)
    model.to(device)
    logger.info('Model Architecture: \n' +  str(model))
    wo_class_error = False

    # data_loader = make_data_loader(
    #     cfg,
    #     logger,
    #     is_train=True,
    #     is_distributed=args.distributed,
    #     start_iter=0)

    data_loaders_val = make_data_loader(
        cfg, logger, is_train=False, is_distributed=args.distributed)
    data_loader_val = data_loaders_val[0]
    
    save_to_disk = utils.get_rank() == 0

    # if args.distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, 
    #         broadcast_buffers=True,
    #         find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS)

    checkpointer = DetectronCheckpointer(
        cfg, model, save_dir=args.output_dir, logger=logger)

    # loading weight
    # TODO: add custom arguments for model weights
    # checkpointer.load(try_to_find(cfg.MODEL.WEIGHT))
    # checkpointer.load(args.weight)
    checkpoint = torch.load(args.weight, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    logger.info(f"{load_res}")
    _ = model.eval()
    
    _result = do_inference(
        logger = logger,
        model = model,
        data_loader = data_loader_val,
        dataset_name=cfg.DATASETS.TEST,
        device=device,
        expected_results=cfg.TEST.EXPECTED_RESULTS,
        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        output_folder=None,
        cfg=cfg,
        verbose=True)

    if utils.is_main_process():
        eval_result = _result[0].results['bbox']['AP']


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GroundingDINO evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
