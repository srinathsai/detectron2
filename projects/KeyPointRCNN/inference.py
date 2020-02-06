import cv2
import os
import argparse
import numpy as np

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# predictor returns bboxes, classes and pred keypoints
# pred keypoints has shape (N, K, 3): N = num instances, K = num keypoints (17), the 3 is
# (x, y, visibility/score/confidence)
# TODO write code to extract keypoints of largest + central bounding box person (like in DensePose) and save as npy + save vis


def get_largest_centred_bounding_box(bboxes, orig_w, orig_h):
    """
    Args:
        bboxes: (N, 4) array of x1 y1 x2 y2 bounding boxes.

    Returns:
        Index of largest and roughtly centred bounding box.
    """
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    sorted_bbox_indices = np.argsort(bboxes_area)[::-1]
    bbox_found = False
    i = 0
    while not bbox_found:
        bbox_index = sorted_bbox_indices[i]
        bbox = bboxes[bbox_index]
        bbox_centre = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        if abs(bbox_centre[0] - orig_w / 2.0) < 50 and abs(bbox_centre[1] - orig_h / 2.0) < 50:
            largest_bbox_index = bbox_index
            bbox_found = True
        i += 1

    return largest_bbox_index


def predict_on_folder(in_folder, out_folder, config_file):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(config_file))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    predictor = DefaultPredictor(cfg)

    image_fnames = [f for f in sorted(os.listdir(in_folder)) if f.endswith('.png')]

    for fname in image_fnames:
        image = cv2.imread(os.path.join(in_folder, fname))
        orig_h, orig_w = image.shape[:2]
        outputs = predictor(image)
        bboxes = outputs['instances'].pred_boxes.cpu().numpy()
        largest_bbox_index = get_largest_centred_bounding_box(bboxes, orig_w, orig_h)
        bbox = bboxes[largest_bbox_index]
        keypoints = outputs['instances'].pred_keypooints[largest_bbox_index].cpu().numpy()
        print(bbox.shape, keypoints.shape)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_folder', type=str)
    parser.add_argument('--config_file', type=str,
                        help='config file name relative to detectron2 configs directory')
    parser.add_argument('--out_folder', type=str)
    parser.add_argument('--gpu', type=str)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.out_folder == 'dataset':
        out_folder = args.in_folder.replace('cropped_frames', 'keypoint_rcnn_results')
    else:
        out_folder = args.out_folder

    predict_on_folder(args.in_folder, out_folder, args.config_file)