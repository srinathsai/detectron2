"""
Script that visualises and saves densepose results generated by apply_net.py in dump mode.
Only visualises for the largest person (largest bounding box) in the image.
Saves the I_image corresponding to each prediction as png images.
"""

import sys
import os
import pickle
import argparse
import numpy as np
import cv2

from densepose.structures import DensePoseResult
sys.path.append("/data/cvfs/as2562/detectron2/projects/DensePose/")
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def apply_colormap(image, vmin=None, vmax=None, cmap='viridis', cmap_seed=1):
    """
    Apply a matplotlib colormap to an image.

    This method will preserve the exact image size. `cmap` can be either a
    matplotlib colormap name, a discrete number, or a colormap instance. If it
    is a number, a discrete colormap will be generated based on the HSV
    colorspace. The permutation of colors is random and can be controlled with
    the `cmap_seed`. The state of the RNG is preserved.
    """
    image = image.astype("float64")  # Returns a copy.
    # Normalization.
    if vmin is not None:
        imin = float(vmin)
        image = np.clip(image, vmin, sys.float_info.max)
    else:
        imin = np.min(image)
    if vmax is not None:
        imax = float(vmax)
        image = np.clip(image, -sys.float_info.max, vmax)
    else:
        imax = np.max(image)
    image -= imin
    image /= (imax - imin)
    # Visualization.
    cmap_ = plt.get_cmap(cmap)
    vis = cmap_(image, bytes=True)
    return vis


def visualise_denspose_results(dump_file, out_folder):
    with open(dump_file, 'rb') as f_results:
        data = pickle.load(f_results)

    # Loop through frames
    for entry in data:
        frame_fname = entry['file_name']
        print(frame_fname)
        if out_folder == 'dataset':
            out_vis_path = frame_fname.replace('cropped_frames', 'densepose_vis')
            out_mask_path = frame_fname.replace('cropped_frames', 'densepose_masks')
        else:
            raise NotImplementedError

        if not os.path.exists(os.path.dirname(out_vis_path)):
            os.makedirs(os.path.dirname(out_vis_path))
            os.makedirs(os.path.dirname(out_mask_path))

        frame = cv2.imread(frame_fname)
        frame = frame.astype(np.float32)
        orig_h, orig_w = frame.shape[:2]

        # Choose the result instance (index) with largest bounding box
        bboxes_xyxy = entry['pred_boxes_XYXY'].numpy()
        bboxes_area = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) \
                      * (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1])
        largest_bbox_index = np.argmax(bboxes_area)

        result_encoded = entry['pred_densepose'].results[largest_bbox_index]
        iuv_arr = DensePoseResult.decode_png_data(*result_encoded)

        # Round bbox to int
        largest_bbox = bboxes_xyxy[largest_bbox_index]
        w1 = largest_bbox[0]
        w2 = largest_bbox[0] + iuv_arr.shape[2]
        h1 = largest_bbox[1]
        h2 = largest_bbox[1] + iuv_arr.shape[1]

        I_image = np.zeros((orig_h, orig_w))
        I_image[int(h1):int(h2), int(w1):int(w2)] = iuv_arr[0, :, :]
        I_image = I_image.astype(np.float32)
        # U_image = np.zeros((orig_h, orig_w))
        # U_image[int(h1):int(h2), int(w1):int(w2)] = iuv_arr[1, :, :]
        # V_image = np.zeros((orig_h, orig_w))
        # V_image[int(h1):int(h2), int(w1):int(w2)] = iuv_arr[2, :, :]

        # Save visualisation and I image (i.e. segmentation mask)
        vis_I_image = apply_colormap(I_image, vmin=0, vmax=24)
        vis_I_image = vis_I_image[:, :, :3]
        plt.imshow(vis_I_image)
        plt.show()
        # overlay = cv2.addWeighted(frame,
        #                           1.0,
        #                           128.0 + 128.0 * np.tile(I_image[:, :, None]/24.0,
        #                                           [1, 1, 3]),
        #                           0.5,
        #                           gamma=0)
        # cv2.imwrite(out_vis_path, overlay)
        # cv2.imwrite(out_mask_path, I_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_file', type=str)
    parser.add_argument('--out_folder', type=str)
    args = parser.parse_args()

    visualise_denspose_results(args.dump_file, args.out_folder)
