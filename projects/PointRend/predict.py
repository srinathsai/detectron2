# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
PointRend Prediction Script.

"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    DatasetEvaluators,
    verify_results,
)

from point_rend import add_pointrend_config



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # model = Trainer.build_model(cfg)
    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     cfg.MODEL.WEIGHTS, resume=args.resume
    # )
    pred = DefaultPredictor(cfg)
    # inputs = cv2.imread("input.jpg")
    # outputs = pred(inputs)
    test_input = torch.ones((224, 224, 3))
    test_input = test_input.cuda().float()
    print(test_input)
    test_output = pred(test_input)
    # res = Trainer.test(cfg, model)
    # if comm.is_main_process():
    #     verify_results(cfg, res)
    # return res



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
