"""
@Brief:
    calc stats for a foldername/
    modified from official inception score implementation
    [bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)
@Author: lzhbrian (https://lzhbrian.me)
@Date: 2019.4.7
@Usage:
    python precalc_stats_official_tf.py foldername/ output_path/
    python precalc_stats_official_tf.py /data4/linziheng/datasets/imagenet/valid_64x64/ imagenet_valid_stats_test.npz
"""

import sys
import os
from glob import glob
import numpy as np
import fid_official_tf
from scipy.misc import imread
import tensorflow as tf

########
# PATHS
########
# data_path = 'data' # set path to training set images
# output_path = 'fid_stats.npz' # path for where to store the statistics

data_path = sys.argv[1]
output_path = sys.argv[2]

# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you

cur_dirname = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = '%s/res/' % cur_dirname

inception_path = '%s/' % MODEL_DIR
print("check for inception model..")
inception_path = fid_official_tf.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

# loads all images into memory (this might require a lot of RAM!)
print("load images..")
image_list = []
for ext in ('*.png', '*.jpg', '*.jpeg', '.bmp'):
    image_list.extend(glob(os.path.join(data_path, ext)))

images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
print("%d images found and loaded" % len(images))


print("create inception graph..")
fid_official_tf.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")


print("calculate FID stats..")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid_official_tf.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")
