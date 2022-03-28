import os
import numpy as np
from tqdm import tqdm


def split_sar_optic(root):
    save_image = os.path.join(root, 'label')
    image_list = os.listdir(save_image)

    np.random.shuffle(image_list)

    train_list = image_list[::2]
    valid_list = [x for x in image_list if x not in train_list]

    with open(os.path.join(root, 'trainM_list.txt'), 'w', encoding='utf-8') as f:
        for each in tqdm(train_list):
            f.write(root + '/GF2/' + os.path.splitext(each)[0] + '.jpg' + ' '
                    + root + '/GF3/' + os.path.splitext(each)[0] + '.jpg' + ' '
                    + root + '/label/' + os.path.splitext(each)[0] + '.png' + '\n')

    with open(os.path.join(root, 'valM_list.txt'), 'w', encoding='utf-8') as f:
        for each in tqdm(valid_list):
            f.write(root + '/GF2/' + os.path.splitext(each)[0] + '.jpg' + ' '
                    + root + '/GF3/' + os.path.splitext(each)[0] + '.jpg' + ' '
                    + root + '/label/' + os.path.splitext(each)[0] + '.png' + '\n')


if __name__ == '__main__':
    root = '/workspace/MaShibin/DATA/korea/cloud'
    split_sar_optic(root)

