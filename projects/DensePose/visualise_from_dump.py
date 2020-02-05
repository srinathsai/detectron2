import sys
sys.path.append("/data/cvfs/as2562/detectron2/projects/DensePose/")
import pickle
import argparse
from densepose.structures import DensePoseResult


def visualise_denspose_results(dump_file):
    with open(dump_file, 'rb') as f_results:
        data = pickle.load(f_results)

    # Loop through frames
    for entry in data:
        frame_fname = entry['file_name']
        print(frame_fname)
        bboxes_xyxy = entry['pred_boxes_XYXY'].numpy()
        # Choose the result instance (index) with largest bounding box
        print(bboxes_xyxy)
        bboxes_area = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) \
                      * (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1])
        print(bboxes_area)
        # results_encoded = entry['pred_densepose']
        # iuv_arr = DensePoseResult.decode_png_data(*result_encoded)
        # print(iuv_arr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_file', type=str)
    args = parser.parse_args()

    visualise_denspose_results(args.dump_file)
