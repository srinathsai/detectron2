import pickle
import argparse


def visualise_denspose_results(dump_file):
    with open(dump_file, 'rb') as f_results:
        data = pickle.load(f_results)

    for entry in data:
        fname = entry['filename']
        result_encoded = entry['pred_densepose']
        print(result_encoded)
        # iuv_arr = DensePoseResult.decode_png_data(*result_encoded)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_file', type=str)
    args = parser.parse_args()

    visualise_denspose_results(args.dump_file)
