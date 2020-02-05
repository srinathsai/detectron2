# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
PointRend Prediction Script.

"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import argparse
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
    pred = DefaultPredictor(cfg)

    input_folder = args.input_folder
    output_masks_folder = input_folder.replace('cropped_frames', 'pointrend_R50FPN_masks')
    output_vis_folder = input_folder.replace('cropped_frames', 'pointrend_R50FPN_vis')
    print("Saving to:", output_masks_folder)
    os.makedirs(output_masks_folder, exist_ok=True)
    os.makedirs(output_vis_folder, exist_ok=True)

    image_fnames = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.png')]
    for fname in image_fnames:
        print(fname)
        input = cv2.imread(os.path.join(input_folder, fname))
        outputs = pred(input)['instances']
        classes = outputs.pred_classes
        masks = outputs.pred_masks
        human_masks = masks[classes == 0]
        human_masks = human_masks.cpu().detach().numpy()
        largest_sum_mask_index = np.argmax(np.sum(human_masks, axis=(1, 2)), axis=0)
        human_mask = human_masks[largest_sum_mask_index, :, :].astype(np.uint8)
        overlay = cv2.addWeighted(input, 1.0,
                                  255 * np.tile(human_mask[:, :, None], [1, 1, 3]),
                                  0.5, gamma=0)
        save_vis_path = os.path.join(output_vis_folder, fname)
        save_mask_path = os.path.join(output_masks_folder, fname)
        cv2.imwrite(save_vis_path, overlay)
        cv2.imwrite(save_mask_path, human_mask)


if __name__ == "__main__":
    # args = default_argument_parser().parse_args()
    # print("Command Line Args:", args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str)
    parser.add_argument("--config_file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)

