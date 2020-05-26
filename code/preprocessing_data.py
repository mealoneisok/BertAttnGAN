import pickle
import os
import numpy as np
from PIL import Image

path_raw = r'CUB\raw'
path_processed = r'CUB\processed'
path_cls = path_raw + r'\classes.txt'
path_files = path_raw + r'\files.txt'
path_bboxes = path_raw + r'\bboxes.txt'
path_raw_text = path_raw + r'\text'
path_processed_text = path_processed + r'\text'
path_processed_img =  path_processed + r'\images'

def crop_bird(path_img, bbox):
    img = Image.open(path_img).convert('RGB')
    w, h = img.size
    r = int(max(bbox[2], bbox[3]) * 0.75)
    cen_x = int((2 * bbox[0] + bbox[2]) / 2)
    cen_y = int((2 * bbox[1] + bbox[3]) / 2)
    y1 = max(0, cen_y - r)
    y2 = min(h, cen_y + r)
    x1 = max(0, cen_x - r)
    x2 = min(w, cen_x + r)
    img = img.crop([x1, y1, x2, y2])
    return img

if __name__ == '__main__':
    with open(path_files, 'r') as f:
        filename_lst = [line[:-1] for line in f]
        filename_lst = [filename.split()[1] for filename in filename_lst]
        filename_lst = [filename[:-4] for filename in filename_lst]

    with open(path_cls, 'r') as f:
        cls_lst = [line[:-1] for line in f]
        cls_lst = [cls.split()[1] for cls in cls_lst]

    #for CUB only
    with open(path_bboxes, 'r') as f:
        bboxes_lst = [line[:-1] for line in f]
        bboxes_lst = [bbox.split()[1:] for bbox in bboxes_lst]
        bboxes_lst = [[int(float(x)) for x in bbox] for bbox in bboxes_lst]

    for cls in cls_lst:
        path_to_add = path_processed_img + '\\' + cls
        if not os.path.exists(path_to_add):
            os.mkdir(path_to_add)

    for filename, bbox in zip(filename_lst, bboxes_lst):
        path_to_load = path_raw + '\\images\\' + filename + '.jpg'
        path_to_save = path_processed_img + '\\' + filename + '.jpg'
        res = crop_bird(path_to_load, bbox)
        res.save(path_to_save)

    rng_filename_idx = np.random.permutation(len(filename_lst))
    train_idx = rng_filename_idx[:int(len(filename_lst) * 0.9)] #10609
    test_idx = rng_filename_idx[int(len(filename_lst) * 0.9):]  #1179
    train_idx = sorted(train_idx)
    test_idx = sorted(test_idx)
    train_id = np.array(filename_lst)[train_idx].tolist()
    test_id = np.array(filename_lst)[test_idx].tolist()

    with open(path_processed + r'\train_idx.pkl', 'wb') as f:
        pickle.dump(train_idx, f)
    with open(path_processed + r'\test_idx.pkl', 'wb') as f:
        pickle.dump(test_idx, f)

    with open(path_processed + r'\train_id.pkl', 'wb') as f:
        pickle.dump(train_id, f)
    with open(path_processed + r'\test_id.pkl', 'wb') as f:
        pickle.dump(test_id, f)

    caps_text = []
    for filename in filename_lst:
        path_cap = path_raw_text + '\\' + filename + '.txt'
        with open(path_cap, 'r') as f:
            caps = []
            for line in f:
                if line[-1] == '\n':
                    cap = line[:-1]
                else:
                    cap = line

    caps_text_train = np.array(caps_text)[train_idx].tolist()
    caps_text_test = np.array(caps_text)[test_idx].tolist()

    with open(path_processed + r'\caps_text_train.pkl', 'wb') as f:
        pickle.dump(caps_text_train, f)
    with open(path_processed + r'\caps_text_test.pkl', 'wb') as f:
        pickle.dump(caps_text_test, f)

