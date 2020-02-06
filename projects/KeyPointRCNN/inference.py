import cv2
import os
import argparse

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
        outputs = predictor(image)
        bboxes = outputs['instances'].pred_boxes
        print(bboxes)




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
