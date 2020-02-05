import pickle


with open('/data/cvfs/as2562/datasets/sports_videos_smpl/class_videos/beach_handball/densepose_results/002/clip_002/person_002/densepose_results.pkl', 'rb') as f_results:
    data = pickle.load(f_results)

print(data)