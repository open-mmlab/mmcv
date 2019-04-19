import lmdb
import os.path as osp
from mmcv.image.io import imfrombytes 


def check_image_valid(img_bytes, flag='color'):
    if img_bytes is None:
        return False
    img = imfrombytes(img_bytes, flag=flag)
    h, w = img.shape[0], img.shape[1]
    if h * w == 0:
        return False
    return True


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(str(k).encode(), v)


def create_rawimage_dataset(output_path, img_file_list, image_tmpl=None, flag='color', check_valid=True):
    """ Create LMDB dataset from a bunch of raw image (within a same video)

    Args:
        output (str): The name of LMDB output path
        img_file_list (list): A list of image files
        flag (str): 'color' or grayscale
        check_valid (bool): Whether to check if every image is valid
    """
    num_samples = len(img_file_list)
    env = lmdb.open(output_path, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i, img_path in enumerate(img_file_list):
        if not osp.exists(img_path):
            print("{} does not exist".format(img_path))
            continue
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        if check_valid:
            if not check_image_valid(img_bytes, flag=flag):
                print('{} is not a valid image'.format(img_path))
                continue

        img_key = osp.splitext(osp.basename(img_path))[0] if image_tmpl is None else image_tmpl.format(cnt)
        cache[img_key] = img_bytes
        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
            print("{} / {} images written.".format(cnt, num_samples))
        cnt += 1
    cache['num_samples'] = str(cnt - 1).encode()
    write_cache(env, cache)
    print('Create dataset with {} samples'.format(num_samples))

