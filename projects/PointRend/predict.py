# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
PointRend Prediction Script.

"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
    inputs = cv2.imread("test.png")
    outputs = pred(inputs)
    print(outputs.keys())
    print(outputs['instances'])
    print(outputs['instances'].pred_masks.shape)
    print(outputs['instances'].pred_classes)
    mask = outputs['instances'].pred_masks[0]
    mask = mask.cpu().detach().numpy()
    plt.imshow(mask.astype(np.uint8))
    plt.show()
    # test_input = torch.ones((224, 224, 3))
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
