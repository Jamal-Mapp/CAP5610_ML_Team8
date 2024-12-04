from pathlib import Path
import json

import numpy as np
import pandas as pd

#process training data
ASL_DIR = '~/workspace/SignLanguage/train'

face_landmarks = {
    'outer_lip': np.array([0, 8, 9, 10, 63, 17, 43, 30, 62, 27, 5, 89, 123, 92, 105, 79, 124, 72, 71, 70, 0]),
    'inner_lip': np.array([3, 26, 25, 24, 64, 23, 32, 29, 61, 28, 4, 90, 122, 91, 94, 85, 125, 86, 87, 88, 3]),
    'right_eye': np.array([1, 7, 66, 55, 54, 53, 52, 51, 59, 39, 50, 49, 48, 42, 41, 57, 1]),
    'left_eye': np.array([67, 69, 127, 116, 115, 114, 113, 112, 120, 101, 111, 110, 109, 104, 103, 118, 67]),
    'lower_right_eyebrow': np.array([11, 13, 12, 19, 15]),
    'upper_right_eyebrow': np.array([22, 18, 34, 20, 35]),
    'lower_left_eyebrow': np.array([73, 75, 74, 81, 77]),
    'upper_left_eyebrow': np.array([84, 80, 96, 82, 97])
}
hand_landmarks = {
    'thumb': np.array([0,1,2,3,4]),
    'index': np.array([0,5,6,7,8]),
    'middle': np.array([0,9,10,11,12]),
    'ring': np.array([0,13,14,15,16]),
    'pinky': np.array([0,17,18,19,20])
}
pose_landmarks = {
    'wrist_to_wrist': np.array([16,14,12,11,13,15,])
}

face_indices = np.unique(np.concatenate(list(face_landmarks.values())))
hand_indices = np.unique(np.concatenate(list(hand_landmarks.values())))
pose_indices = np.unique(np.concatenate(list(pose_landmarks.values())))

indices = np.concatenate([
    face_indices,          # face
    hand_indices+468,      # left hand
    pose_indices+468+21,   # pose
    hand_indices+468+21+33 # right hand
])

np.save('indices.npy', indices)

ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def load_impt_data(pq_path):
    data = load_relevant_data_subset(pq_path)
    impt = data[:,indices].astype(np.float16)
    del data
    return impt

signs_df = pd.read_csv(ASL_DIR+'/train.csv')

sign_to_label = pd.read_json(ASL_DIR+ '/sign_to_prediction_index_map.json', typ='series', orient='index')
signs_df['label'] = signs_df.sign.map(sign_to_label)

signs_df.to_csv('train.csv', index=False)

data = {}
for path in signs_df.path:
    out = Path(path)
    data[out.stem] = load_impt_data(ASL_DIR +'/'+ path)

np.savez('data.npz', **data)

landmarks = dict()
for t in [
    (face_landmarks, 0, 'face'),
    (hand_landmarks, 468, 'left_hand'),
    (pose_landmarks, 468+21, 'pose'),
    (hand_landmarks, 468+21+33, 'right_hand')]:
    for k,l in t[0].items():
        name = t[2]+'_'+k
        mask = np.expand_dims(indices, 1) == (l+t[1])
        zeros = np.zeros(mask.shape)
        zeros[:] = np.expand_dims(np.arange(len(indices)), 1)
        landmarks[name] = np.max((zeros * mask), axis=0).astype(int).tolist()

with open('landmarks.json', 'w') as f:
    json.dump(landmarks, f)

# process test data
ASL_DIR = '~/workspace/SignLanguage/test'

np.save('test_indices.npy', indices)

signs_df = pd.read_parquet(ASL_DIR+'/labels.parqet')
signs_df['sequence_id'] = signs_df['path'].str.split('/').str[-1].str.split('.').str[0]

sign_to_label = pd.read_json(ASL_DIR+ '/sign_to_prediction_index_map.json', typ='series', orient='index')

signs_df['label'] = signs_df.sign.map(sign_to_label)

signs_df.to_csv('test.csv', index=False)
data = {}
for path in signs_df.path:
    out = Path(path)
    data[out.stem] = load_impt_data(ASL_DIR +'/'+ path)

np.savez('test_data.npz', **data)
