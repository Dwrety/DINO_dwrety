import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import logging
import datetime
import torch 
import torch.distributed as dist

import util.misc as utils
from util.utils import slprint, to_device, ModelEma

from data.datasets.evaluation import evaluate

from .amp import autocast, GradScaler
from .metric_logger import MetricLogger
from .inference import do_inference


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = utils.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
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
        val_data_loader=None,
        meters=None,
        zero_shot=False):

    logger.info("Start training")
    model.train()

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    
    model_ema = None
    if cfg.SOLVER.MODEL_EMA > 0:
        model_ema = ModelEma(model, decay=cfg.MODEL_EMA)
        
    start_training_time = time.time()
    end = time.time()

    if cfg.SOLVER.USE_AMP:
        scaler = GradScaler()

    global_rank = utils.get_rank()

    if cfg.SOLVER.CHECKPOINT_PER_EPOCH != -1 and cfg.SOLVER.MAX_EPOCH >= 1:
        checkpoint_period = len(data_loader) * cfg.SOLVER.CHECKPOINT_PER_EPOCH // cfg.SOLVER.MAX_EPOCH
    
    if global_rank <= 0 and cfg.SOLVER.MAX_EPOCH >= 1:
        logger.info(f"Iter per epoch {len(data_loader) // cfg.SOLVER.MAX_EPOCH}")

    # Adapt the weight decay
    if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        milestone_target = 0
        for i, milstone in enumerate(list(scheduler.milestones)):
            if scheduler.last_epoch >= milstone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
                milestone_target = i+1

    for iteration, (images, targets, idxs, positive_map, positive_map_eval, greenlight_map) in enumerate(
        data_loader, start_iter):
        nnegative = sum(len(target) < 1 for target in targets)
        nsample = len(targets)
        if nsample == nnegative or nnegative > nsample * cfg.SOLVER.MAX_NEG_PER_BATCH:
            logger.info('[WARNING] Sampled {} negative in {} in a batch, greater the allowed ratio {}, skip'.
                        format(nnegative, nsample, cfg.SOLVER.MAX_NEG_PER_BATCH))
            continue

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        captions = None
        try:
            targets = [target.to(device) for target in targets]
            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
        except:
            pass

        # freeze clip backbone is settled in groundingdino.py
        # # Freeze language backbone
        # if cfg.MODEL.LANGUAGE_BACKBONE.FREEZE:
        #     if hasattr(model, "module"):
        #         model.module.language_backbone.eval()
        #     else:
        #         model.language_backbone.eval()

        # TODO: test AMP solver
        # TODO: test pytorch checkpointing
        '''
                    outputs is a dict containing 4 fields:
                        pred_logits:    Tensor of shape [batch_size, num_obj_queries, max_seq_length]
                        pred_boxes:     Tensor of shape [batch_size, num_obj_queries, 4]
                        interm_outputs: Dict containing 2 fields, basically the 1st stage of two-stage DETR.
                                            pred_logits:    Tensor of shape [batch_size, num_obj_queries, max_seq_length]
                                            pred_boxes:     Tensor of shape [batch_size, num_obj_queries, 4]
                        interm_outputs_for_matching_pre: Dict containing 2 fields, basically the 1st stage of two-stage DETR.
                                            pred_logits:    Tensor of shape [batch_size, num_obj_queries, max_seq_length]
                                            pred_boxes:     Tensor of shape [batch_size, num_obj_queries, 4]
        '''
        if cfg.SOLVER.USE_AMP:
            with autocast():
                outputs = model(images, targets)
                # TODO: add matcher and criterion
                # if len(captions) > 0:
                #     outputs = model(images, targets, positive_map=positive_map, greenlight_map=greenlight_map)
                # else:
                #     outputs = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses) or torch.isinf(losses):
                logging.error("NaN encountered, ignoring")
                losses[losses != losses] = 0
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            if len(captions) > 0:
                loss_dict = model(images, targets, captions, positive_map)
            else:
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses) or torch.isinf(losses):
                losses[losses != losses] = 0
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

        # TODO: Adapt the weight decay: only support multiStepLR
        # if cfg.SOLVER.WEIGHT_DECAY_SCHEDULE and hasattr(scheduler, 'milestones'):
        #     if milestone_target < len(scheduler.milestones):
        #         next_milestone = list(scheduler.milestones)[milestone_target]
        #     else:
        #         next_milestone = float('inf')
        #     if scheduler.last_epoch >= next_milestone * cfg.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO:
        #         gamma = scheduler.gamma
        #         logger.info("Drop the weight decay by {}!".format(gamma))
        #         for param in optimizer.param_groups:
        #             if 'weight_decay' in param:
        #                 param['weight_decay'] *= gamma
        #         # move the target forward
        #         milestone_target += 1

        # TODO: reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = reduce_loss_dict(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        # meters.update(loss=losses_reduced, **loss_dict_reduced)
        # if model_ema is not None:
        #     model_ema.update(model)
        #     arguments["model_ema"] = model_ema.state_dict()

        # batch_time = time.time() - end
        # end = time.time()
        # meters.update(time=batch_time, data=data_time)
        # eta_seconds = meters.time.global_avg * (max_iter - iteration)
        # eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # if iteration % 20 == 0 or iteration == max_iter:
        # TODO: DEBUG  if iteration % 1 == 0 or iteration == max_iter:
        #     #logger.info(
        #     if global_rank <= 0:
        #         # print(
        #         logger.info(
        #             meters.delimiter.join(
        #                 [
        #                     "eta: {eta}",
        #                     "iter: {iter}",
        #                     "{meters}",
        #                     "lr: {lr:.6f}",
        #                     "wd: {wd:.6f}",
        #                     "max mem: {memory:.0f}",
        #                 ]
        #             ).format(
        #                 eta=eta_string,
        #                 iter=iteration,
        #                 meters=str(meters),
        #                 lr=optimizer.param_groups[0]["lr"],
        #                 wd=optimizer.param_groups[0]["weight_decay"],
        #                 memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
        #             )
        #         )

        # if val_data_loader and (iteration % checkpoint_period == 0 or iteration == max_iter):
        if val_data_loader:
            if utils.is_main_process():
                print("Evaluating")
            eval_result = 0.0
            model.eval()
            if cfg.SOLVER.TEST_WITH_INFERENCE:
                with torch.no_grad():
                    try:
                        _model = model.module
                    except:
                        _model = model
                    _result = do_inference(
                        logger = logger,
                        model = _model,
                        data_loader = val_data_loader,
                        dataset_name="val",
                        device=device,
                        expected_results=cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                        output_folder=None,
                        cfg=cfg,
                        verbose=False
                    )
                    if utils.is_main_process():
                        eval_result = _result[0].results['bbox']['AP']
            else:
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, *_ = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model(images)
                        else:
                            # captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            # output = model(images, captions, positive_map)
                            outputs = model(images, targets)

                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = utils.all_gather(results_dict)
                if utils.is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                            box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
            model.train()

            if model_ema is not None and cfg.SOLVER.USE_EMA_FOR_MONITOR:
                model_ema.ema.eval()
                results_dict = {}
                cpu_device = torch.device("cpu")
                for i, batch in enumerate(val_data_loader):
                    images, targets, image_ids, positive_map, positive_map_eval = batch
                    with torch.no_grad():
                        images = images.to(device)
                        if positive_map is None:
                            output = model_ema.ema(images)
                        else:
                            captions = [t.get_field("caption") for t in targets if "caption" in t.fields()]
                            output = model_ema.ema(images, captions, positive_map)
                        output = [o.to(cpu_device) for o in output]
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                all_predictions = utils.all_gather(results_dict)
                if utils.is_main_process():
                    predictions = {}
                    for p in all_predictions:
                        predictions.update(p)
                    predictions = [predictions[i] for i in list(sorted(predictions.keys()))]
                    eval_result, _ = evaluate(val_data_loader.dataset, predictions, output_folder=None,
                                              box_only=cfg.DATASETS.CLASS_AGNOSTIC)
                    if cfg.DATASETS.CLASS_AGNOSTIC:
                        eval_result = eval_result.results['box_proposal']['AR@100']
                    else:
                        eval_result = eval_result.results['bbox']['AP']
                
            arguments.update(eval_result=eval_result)

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
            break

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )