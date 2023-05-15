import logging

from .vg_eval import do_vg_evaluation


def vg_evaluation(logger, dataset, predictions, output_folder, box_only, eval_attributes=False, **_):
    logger.info("performing vg evaluation, ignored iou_types.")
    return do_vg_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        box_only=box_only,
        eval_attributes=eval_attributes,
        logger=logger,
    )
