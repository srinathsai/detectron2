import pickle





def visualise_denspose_results(dump_file):
    with open(dump_file) as f_results:
        data = pickle.load(f_results)

    for entry in data:
        fname = entry['filename']
        result_encoded = entry['pred_densepose']
        print(result_encoded)
        # iuv_arr = DensePoseResult.decode_png_data(*result_encoded)