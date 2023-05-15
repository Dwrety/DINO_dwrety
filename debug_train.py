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
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils
import data
from data import build_dataset, make_data_loader, try_to_find
from engine import make_optimizer, make_lr_scheduler, MetricLogger, DetectronCheckpointer, do_train


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
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
    parser.add_argument('--output_dir', default='',
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
    
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(
        output=os.path.join(args.output_dir, 'info.txt'), 
        distributed_rank=args.rank, 
        color=True, 
        name="groundingdino")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))

    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))

    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    # logger.info("args: " + str(args) + '\n')

    device = torch.device(args.device)

    # # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    # model, criterion, postprocessors = build_model_main(args)
    model = build_model_main(cfg)
    model.to(device)
    logger.info('Model Architecture: \n' +  str(model))
    wo_class_error = False

    data_loader = make_data_loader(
        cfg,
        logger,
        is_train=True,
        is_distributed=args.distributed,
        start_iter=0)

    if cfg.TEST.DURING_TRAINING: 
        data_loaders_val = make_data_loader(
            cfg, logger, is_train=False, is_distributed=args.distributed)
        data_loaders_val = data_loaders_val[0]
    else: 
        data_loaders_val = None
    
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    save_to_disk = utils.get_rank() == 0

    # model_without_ddp = model
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            broadcast_buffers=True,
            find_unused_parameters=cfg.SOLVER.FIND_UNUSED_PARAMETERS
        )
        # model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))
    
    arguments = {}
    arguments["iteration"] = 0

    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, args.output_dir, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load(try_to_find(cfg.MODEL.WEIGHT))
    arguments.update(extra_checkpoint_data)

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    meters = MetricLogger(delimiter="  ")
    
    do_train(
        cfg,
        logger,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        data_loaders_val,
        meters)
    
    # if args.frozen_weights is not None:
    #     checkpoint = torch.load(args.frozen_weights, map_location='cpu')
    #     model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # output_dir = Path(args.output_dir)
    # if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
    #     args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
    # if args.resume:
    #     if args.resume.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.resume, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.resume, map_location='cpu')
    #     model_without_ddp.load_state_dict(checkpoint['model'])
    #     if args.use_ema:
    #         if 'ema_model' in checkpoint:
    #             ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
    #         else:
    #             del ema_m
    #             ema_m = ModelEma(model, args.ema_decay)                

    #     if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         args.start_epoch = checkpoint['epoch'] + 1

    # if (not args.resume) and args.pretrain_model_path:
    #     checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
    #     from collections import OrderedDict
    #     _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
    #     ignorelist = []

    #     def check_keep(keyname, ignorekeywordlist):
    #         for keyword in ignorekeywordlist:
    #             if keyword in keyname:
    #                 ignorelist.append(keyname)
    #                 return False
    #         return True

    #     logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
    #     _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})

    #     _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
    #     logger.info(str(_load_output))

    #     if args.use_ema:
    #         if 'ema_model' in checkpoint:
    #             ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
    #         else:
    #             del ema_m
    #             ema_m = ModelEma(model, args.ema_decay)        


    # if args.eval:
    #     os.environ['EVAL_FLAG'] = 'TRUE'
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
    #     if args.output_dir:
    #         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

    #     log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
    #     if args.output_dir and utils.is_main_process():
    #         with (output_dir / "log.txt").open("a") as f:
    #             f.write(json.dumps(log_stats) + "\n")

    #     return

    # print("Start training")
    # start_time = time.time()
    # best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    # for epoch in range(args.start_epoch, args.epochs):
        # epoch_start_time = time.time()
        # if args.distributed:
        #     sampler_train.set_epoch(epoch)
    #     train_stats = train_one_epoch(
    #         model, criterion, data_loader_train, optimizer, device, epoch,
    #         args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
    #     if args.output_dir:
    #         checkpoint_paths = [output_dir / 'checkpoint.pth']

    #     if not args.onecyclelr:
    #         lr_scheduler.step()
    #     if args.output_dir:
    #         checkpoint_paths = [output_dir / 'checkpoint.pth']
    #         # extra checkpoint before LR drop and every 100 epochs
    #         if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
    #             checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
    #         for checkpoint_path in checkpoint_paths:
    #             weights = {
    #                 'model': model_without_ddp.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'lr_scheduler': lr_scheduler.state_dict(),
    #                 'epoch': epoch,
    #                 'args': args,
    #             }
    #             if args.use_ema:
    #                 weights.update({
    #                     'ema_model': ema_m.module.state_dict(),
    #                 })
    #             utils.save_on_master(weights, checkpoint_path)
                
    #     # eval
    #     test_stats, coco_evaluator = evaluate(
    #         model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
    #         wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
    #     )
    #     map_regular = test_stats['coco_eval_bbox'][0]
    #     _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
    #     if _isbest:
    #         checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
    #         utils.save_on_master({
    #             'model': model_without_ddp.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'lr_scheduler': lr_scheduler.state_dict(),
    #             'epoch': epoch,
    #             'args': args,
    #         }, checkpoint_path)
    #     log_stats = {
    #         **{f'train_{k}': v for k, v in train_stats.items()},
    #         **{f'test_{k}': v for k, v in test_stats.items()},
    #     }

    #     # eval ema
    #     if args.use_ema:
    #         ema_test_stats, ema_coco_evaluator = evaluate(
    #             ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
    #             wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
    #         )
    #         log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
    #         map_ema = ema_test_stats['coco_eval_bbox'][0]
    #         _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
    #         if _isbest:
    #             checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
    #             utils.save_on_master({
    #                 'model': ema_m.module.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'lr_scheduler': lr_scheduler.state_dict(),
    #                 'epoch': epoch,
    #                 'args': args,
    #             }, checkpoint_path)
    #     log_stats.update(best_map_holder.summary())

    #     ep_paras = {
    #             'epoch': epoch,
    #             'n_parameters': n_parameters
    #         }
    #     log_stats.update(ep_paras)
    #     try:
    #         log_stats.update({'now_time': str(datetime.datetime.now())})
    #     except:
    #         pass
        
    #     epoch_time = time.time() - epoch_start_time
    #     epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
    #     log_stats['epoch_time'] = epoch_time_str

    #     if args.output_dir and utils.is_main_process():
    #         with (output_dir / "log.txt").open("a") as f:
    #             f.write(json.dumps(log_stats) + "\n")

    #         # for evaluation logs
    #         if coco_evaluator is not None:
    #             (output_dir / 'eval').mkdir(exist_ok=True)
    #             if "bbox" in coco_evaluator.coco_eval:
    #                 filenames = ['latest.pth']
    #                 if epoch % 50 == 0:
    #                     filenames.append(f'{epoch:03}.pth')
    #                 for name in filenames:
    #                     torch.save(coco_evaluator.coco_eval["bbox"].eval,
    #                                output_dir / "eval" / name)
    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))

    # # remove the copied files.
    # copyfilelist = vars(args).get('copyfilelist')
    # if copyfilelist and args.local_rank == 0:
    #     from datasets.data_util import remove
    #     for filename in copyfilelist:
    #         print("Removing: {}".format(filename))
    #         remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GroundingDINO training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
