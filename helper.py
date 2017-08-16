import numpy as np
import pickle as pkl


def write_to_file(data, output_file='./captions.txt'):
    """
    Save generated captions.
    :param data:
    :param output_file:
    :return:
    """
    with open(output_file, 'a') as f:
        np.savetxt(f, data, fmt='%s', delimiter=",")

    # with open(output_file, 'w') as f:
    #     for item in data.tolist():
    #         f.write(str(item) + '\n')


def load_dictionary(loc='/home/yhx/CaptionGAN/data/vse/coco.npz.dictionary.pkl'):
    """
    Load a dictionary
    :param loc:
    :return:
    """
    with open(loc, 'rb') as f:
        word_dict = pkl.load(f)

    return word_dict


def make_ids():
    """Split ids from images filename
    e.g.: fs...0001.jpg --> 0001
    """

    coco_id = np.genfromtxt('/home/yang/Downloads/FILE/ml/GANCapt/input/coco_train.txt',
                            dtype=None,
                            comments=None)
    ids = []
    for x in coco_id:
        xs = x.split('_')[-1].split('.')[0]
        ids.append(int(xs))
    np.savetxt('/home/yang/Downloads/FILE/ml/GANCapt/input/coco_train_ids.txt', ids, fmt='%s')


def npy_to_txt(npy_path):
    """npy to txt"""
    npy_file = np.load(npy_path)
    np.savetxt('./train/coco_train_imgs_embedding.txt', npy_file, fmt='%e')

    print(npy_file.shape)
    print(npy_file)


if __name__ == '__main__':
    npy_to_txt('./train/train_imgs_embedding.npy')
