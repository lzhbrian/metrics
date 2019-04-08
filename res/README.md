# res/

* in this [res/](./) dir, should contain:
	* inception pretrained weights `inception-2015-12-05.tgz` from [link](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz) (for using the TF implementation)

* in [res/stats_tf/](stats_tf/) dir, should contain:
	* precalculated statistics for datasets from [link](http://bioinf.jku.at/research/ttur/)
		* [Cropped CelebA](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_celeba.npz), [LSUN bedroom](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_lsun_train.npz), [CIFAR 10](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz), [SVHN](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_svhn_train.npz), [ImageNet Train](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_imagenet_train.npz), [ImageNet Valid](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_imagenet_valid.npz)

* in [res/stats_pytorch](stats_pytorch/) dir
	* store precalculated stats using [is_fid_pytorch.py](../is_fid_pytorch.py)



