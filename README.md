# metrics

This repo contains information/implementation (PyTorch, Tensorflow) about IS and FID score. This is a handy toolbox that you can easily add to your projects. TF implementations are intended to compute the exact same output as the official ones for reporting in papers. Discussion/PR/Issues are very welcomed.



## Usage

Put this `metrics/` folder in your projects, and __see below (Pytorch), and each .py's head comment__ for usage.

We also need to download some files in [res/](res/), see [res/README.md](res/README.md) for more details.



## TF implementations (almost the same as official, just changed the interface, can be reported in papers)

- [x] [inception_score_official_tf.py](inception_score_official_tf.py): inception score
- [x] [fid_official_tf.py](fid_official_tf.py): FID score
- [x] [precalc_stats_official_tf.py](precalc_stats_official_tf.py): calculate stats (mu, sigma)



## Pytorch Implementation (CANNOT report in papers, but can get an quick view)

* Requirements

    * pytorch, torchvision, scipy, numpy, tqdm
* [is_fid_pytorch.py](is_fid_pytorch.py)
    * [x] inception score, get around `mean=9.67278, std=0.14992` for CIFAR-10 train data when n_split=10
    * [x] FID score
    * [x] calculate stats for custom images in a folder (mu, sigma)
    * [x] multi-GPU support by `nn.DataParallel`
        * e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3` will use 4 GPU.
* command line usage
    * calculate IS, FID
        ```bash
        # calc IS score on CIFAR10, will download CIFAR10 data to ../data/cifar10
        python is_fid_pytorch.py
        
        # calc IS score on custom images in a folder/
        python is_fid_pytorch.py --path foldername/
        
        # calc IS, FID score on custom images in a folder/, compared to CIFAR10 (given precalculated stats)
        python is_fid_pytorch.py --path foldername/ --fid res/stats_pytorch/fid_stats_cifar10_train.npz
        
        # calc FID on custom images in two folders/
        python is_fid_pytorch.py --path foldername1/ --fid foldername2/
        
        # calc FID on two precalculated stats
        python is_fid_pytorch.py --path res/stats_pytorch/fid_stats_cifar10_train.npz --fid res/stats_pytorch/fid_stats_cifar10_train.npz
        ```

    * precalculate stats
        ```bash
        # precalculate stats store as npz for CIFAR 10, will download CIFAR10 data to ../data/cifar10
        python is_fid_pytorch.py --save-stats-path res/stats_pytorch/fid_stats_cifar10_train.npz
        
        # precalculate stats store as npz for images in folder/
        python is_fid_pytorch.py --path foldername/ --save-stats-path res/stats_pytorch/fid_stats_folder.npz
        ```

        

* in code usage

    * `mode=1`: image tensor has already normalized by `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
    * `mode=2`: image tensor has already normalized by `mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500]`
        ```python
        from metrics import is_fid_pytorch
        
        # using precalculated stats (.npz) for FID calculation
        is_fid_model = is_fid_pytorch.ScoreModel(mode=2, stats_file='res/stats_pytorch/fid_stats_cifar10_train.npz', cuda=cuda)
        imgs_nchw = torch.Tensor(50000, C, H, W) # torch.Tensor in -1~1, normalized by mean=[0.500, 0.500, 0.500], std=[0.500, 0.500, 0.500]
        is_mean, is_std, fid = is_fid_model.get_score_image_tensor(imgs_nchw)
        
        # we can also pass in mu, sigma for get_score_image_tensor()
        is_fid_model = is_fid_pytorch.ScoreModel(mode=2, cuda=cuda)
        mu, sigma = is_fid_pytorch.read_stats_file('res/stats_pytorch/fid_stats_cifar10_train.npz')
        is_mean, is_std, fid = is_fid_model.get_score_image_tensor(imgs_nchw, mu1=mu, sigma1=sigma)
        
        # if no need FID
        is_fid_model = is_fid_pytorch.ScoreModel(mode=2, cuda=cuda)
        is_mean, is_std, _ = is_fid_model.get_score_image_tensor(imgs_nchw)
        
        # if want stats (mu, sigma) for imgs_nchw, send in return_stats=True
        is_mean, is_std, _, mu, sigma = is_fid_model.get_score_image_tensor(imgs_nchw, return_stats=True)
        
        # from pytorch dataset, use get_score_dataset(), instead of get_score_image_tensor(), other usage is the same
        cifar = dset.CIFAR10(root='../data/cifar10', download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])
                            )
        IgnoreLabelDataset(cifar)
        is_mean, is_std, _ = is_fid_model.get_score_dataset(IgnoreLabelDataset(cifar))
        ```


## TODO

- [ ] Refactor TF implementation of IS, FID Together
- [ ] MS-SSIM score - PyTorch
- [ ] MS-SSIM score - Tensorflow



## Info

### Inception Score (IS)

* Assumption
    * MEANINGFUL: The generated image should be clear, the output probability of a classifier network should be [0.9, 0.05, ...] (largely skewed to a class). $p(y|\mathbf{x})$ is of __low entropy__.
    * DIVERSITY: If we have 10 classes, the generated image should be averagely distributed. So that the marginal distribution $p(y) = \frac{1}{N} \sum_{i=1}^{N} p(y|\mathbf{x}^{(i)})$ is of __high entropy__.
    * Better models: KL Divergence of $p(y|\mathbf{x})$ and $p(y)$ should be high.
* Formulation
    * $\mathbf{IS} = \exp (\mathbb{E}_{\mathbf{x} \sim p_g} D_{KL} [p(y|\mathbf{x}) || p(y)] )$
    * where
        * $\mathbf{x}$ is sampled from generated data
        * $p(y|\mathbf{x})​$ is the output probability of Inception v3 when input is $\mathbf{x}​$
        * $p(y) = \frac{1}{N} \sum_{i=1}^{N} p(y|\mathbf{x}^{(i)})$ is the average output probability of all generated data (from InceptionV3, 1000-dim vector)
        * $D_{KL} (\mathbf{p}||\mathbf{q}) = \sum_{j} p_{j} \log \frac{p_j}{q_j}$, where $j$ is the dimension of the output probability.

* Explanation
    * $p(y)$ is a evenly distributed vector
    * larger $\mathbf{IS}​$ score -> larger KL divergence -> larger diversity and clearness
* Reference
    * Official TF implementation is in [openai/improved-gan](https://github.com/openai/improved-gan)
    * Pytorch Implementation: [sbarratt/inception-score-pytorch](https://github.com/sbarratt/inception-score-pytorch)
    * TF seemed to provide a [good implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py)
    * [scipy.stats.entropy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)
    * [zhihu: Inception Score 的原理和局限性](https://zhuanlan.zhihu.com/p/54146307)
    * [A Note on the Inception Score](https://arxiv.org/abs/1801.01973)



### Fréchet Inception Distance (FID)

* Formulation
    * $\mathbf{FID} = ||\mu_r - \mu_g||^2 + Tr(\Sigma_{r} + \Sigma_{g} - 2(\Sigma_r \Sigma_g)^{1/2})​$
    * where
        * $Tr$ is [trace of a matrix (wikipedia)](https://en.wikipedia.org/wiki/Trace_(linear_algebra))
        * $X_r \sim \mathcal{N}(\mu_r, \Sigma_r)$ and $X_g \sim \mathcal{N}(\mu_g, \Sigma_g)$ are the 2048-dim activations  the InceptionV3 pool3 layer
        * $\mu_r$ is the mean of real photo's feature
        * $\mu_g$ is the mean of generated photo's feature
        * $\Sigma_r$ is the covariance matrix of real photo's feature
        * $\Sigma_g$ is the covariance matrix of generated photo's feature

* Reference
    * Official TF implementation: [bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR)
    * Pytorch Implementation: [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid)
    * TF seemed to provide a [good implementation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py)
    * [zhihu: Frechet Inception Score (FID)](https://zhuanlan.zhihu.com/p/54213305)
    * [Explanation from Neal Jean](https://nealjean.com/ml/frechet-inception-distance/)

