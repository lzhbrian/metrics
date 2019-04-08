"""
@Brief:
    Tensorflow implementation of inception score, should be the same as the official one
    modified from official inception score implementation
    [openai/improved-gan](https://github.com/openai/improved-gan)
@Author: lzhbrian (https://lzhbrian.me)
@Date: 2019.4.5
@Last Modified: 2019.4.7
@Usage:
    # CMD
        # calculate IS on CIFAR10 train
        python inception_score_official_tf.py

        # calculate IS on custom images in foldername/
        python inception_score_official_tf.py foldername/
        python inception_score_official_tf.py /data4/linziheng/datasets/fashionai-attributes-challenge-baseline/fashionAI_attributes_test/test/Images/coat_length_labels/

    # use it in code
        ```
        from metrics import inception_score_official_tf
        is_mean, is_std = get_inception_score(img_list, splits=10)
        ```
@Note:
    Updated 2019.4.7:
    after checking out http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    I found that this model file has already contained:

    1. The Bilinear Resize Layer
        resizing arbitrary input to 299x299
        ```
        import tensorflow as tf
        import os
        with tf.gfile.FastGFile('classify_image_graph_def.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        sess = tf.Session()
        resize_bilinear = sess.graph.get_tensor_by_name('ResizeBilinear:0')
        sess.run(resize_bilinear, {'ExpandDims:0': np.ones((1, H, W, 3))}).shape # (1, 299, 299, 3)
        ```
        so this code can fit arbitrary input image size

    2. The Normalization Layer
        input: 0~255
        normalization: subtracted by 128, then divided by 128
        output: -1~1
        ```
        import tensorflow as tf
        import os
        with tf.gfile.FastGFile(os.path.join('classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        sess = tf.Session()

        Sub = sess.graph.get_tensor_by_name('Sub:0')
        sess.run(Sub, {'ExpandDims:0': 255 * np.zeros((1,299,299,3))})
        # output is all -128

        Mul = sess.graph.get_tensor_by_name('Mul:0')
        sess.run(Mul, {'ExpandDims:0': 255 * np.zeros((1,299,299,3))})
        # output is all -1
        ```
        so the input image range of this code shall be 0~255


    Results:

    On CIFAR-10 train, n_split=10, tf-1.10:
    without random.shuffle, input is 32x32
        get mean=11.237364, std=0.11623184 (consistent with paper)
    with random.shuffle, input is 32x32
        1) get mean=11.242347, std=0.18466103
        2) get mean=11.237335, std=0.10733857
        3) get mean=11.234492, std=0.17140374

    On Imagenet 64x64, n_split=10, tf-1.10:
    with random.shuffle
        get mean=63.40744, std=1.3286287
"""

import os
import os.path
import sys

import numpy as np
from six.moves import urllib
import tensorflow as tf
import math

from tqdm import tqdm

cur_dirname = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = '%s/res/' % cur_dirname
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None


# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
# numpy array shape should be in H x W x C
def get_inception_score(images, splits=10, bs=256):
    assert(type(images) == list)
    assert(type(images[0]) == np.ndarray)
    assert(len(images[0].shape) == 3)
    assert(np.max(images[0]) > 10)
    assert(np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    with tf.Session() as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))

        print('passing through inception network ...')
        for i in tqdm(range(n_batches)):
            # sys.stdout.write(".")
            # sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]

            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {'InputTensor:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)


# This function is called automatically.
def _init_inception():
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
              filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')

    import tarfile
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Import model with a modification in the input tensor to accept arbitrary batch size.
        input_tensor = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='InputTensor')
        _ = tf.import_graph_def(graph_def, name='', input_map={'ExpandDims:0': input_tensor})
    # Works with an arbitrary minibatch size.
    with tf.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.set_shape(tf.TensorShape(new_shape))
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)


if softmax is None:
    _init_inception()


if __name__ == '__main__':

    import random

    def cal_on_cifar10_train():
        # CIFAR 10 utils
        def maybe_download_and_extract(data_dir, url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
            if not os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                filename = url.split('/')[-1]
                filepath = os.path.join(data_dir, filename)
                if not os.path.exists(filepath):
                    def _progress(count, block_size, total_size):
                        sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                                         float(count * block_size) / float(
                                                                             total_size) * 100.0))
                        sys.stdout.flush()

                    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
                    print()
                    statinfo = os.stat(filepath)
                    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
                    tarfile.open(filepath, 'r:gz').extractall(data_dir)

        def unpickle(file):
            import pickle
            fo = open(file, 'rb')
            d = pickle.load(fo, encoding='latin1')
            fo.close()
            # load as N x H x W x C
            return {'x': d['data'].reshape((len(d['data']), 3, 32, 32)).transpose(0, 2, 3, 1),
                    'y': np.array(d['labels']).astype(np.uint8)}

            # normalized to -1 ~ +1
            # return {'x': np.cast[np.float32]((-127.5 + d['data'].reshape((10000, 3, 32, 32))) / 128.).transpose(0, 2, 3, 1),
            #         'y': np.array(d['labels']).astype(np.uint8)}

        def load(data_dir, subset='train'):
            maybe_download_and_extract(data_dir)
            if subset == 'train':
                train_data = [unpickle(os.path.join(data_dir, 'cifar-10-batches-py/data_batch_' + str(i))) for i in
                              range(1, 6)]
                trainx = np.concatenate([d['x'] for d in train_data], axis=0)
                trainy = np.concatenate([d['y'] for d in train_data], axis=0)
                return trainx, trainy
            elif subset == 'test':
                test_data = unpickle(os.path.join(data_dir, 'cifar-10-batches-py/test_batch'))
                testx = test_data['x']
                testy = test_data['y']
                return testx, testy
            else:
                raise NotImplementedError('subset should be either train or test')

        train_x, train_y = load('%s/../data/cifar10' % cur_dirname, subset='train')
        train_x = list(train_x)
        random.shuffle(train_x)

        # train_x is list of images (shape = H x W x C, val = 0~255)
        is_mean, is_std = get_inception_score(train_x, splits=10)
        print(is_mean, is_std)

    # if no arg, calc cifar10 train IS score
    if len(sys.argv) == 1:
        cal_on_cifar10_train()

    # if argv have foldername, calc IS score of pictures in this folder
    else:
        import scipy.misc

        # read a folder
        foldername = sys.argv[1]

        from glob import glob
        files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '.bmp'):
            files.extend(glob(os.path.join(foldername, ext)))

        img_list = []
        print('reading images ...')
        for file in tqdm(files):
            img = scipy.misc.imread(file, mode='RGB')
            img_list.append(img)
        random.shuffle(img_list)
        is_mean, is_std = get_inception_score(img_list, splits=10)
        print(is_mean, is_std)
