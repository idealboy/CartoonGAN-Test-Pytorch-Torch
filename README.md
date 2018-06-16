# CartoonGAN-Test-Pytorch-Torch
Pytorch and Torch testing code of [CartoonGAN]()http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2205.pdf [Chen et al., CVPR18]. With the released pretrained models by authors, I made these simple scripts for a quick test. The training code should be similar to the popular GAN-based image-translation frameworks and is not included here.


## Getting started

```
git clone https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch
cd CartoonGAN-Test-Pytorch-Torch
```

## Pytorch

The original pretrained models are Torch `nngraph` models, which cannot be loaded in Pytorch through `load_lua`. So I manully copy the weights (bias) layer by layer and convert them to `.pth` models. 

- Download the converted models:

```
sh pretrained_model/download_pth.sh
```

- For testing:

```
python test.py --input_dir YourImgDir --style Hosoda --gpu 0
```

## Torch

Working with original models with Torch is also fine. I just convert the weights (bias) in their models from CudaTensor to FloatTensor so that `cudnn' is not required for loading models.

- Download the models:

```
sh pretrained_model/download_t7.sh
```

- For testing:

```
th test.lua -input_dir YourImgDir -style Hosoda -gpu 0
```

## Examples (Left: input, Right: output)

<p>
    <img src='test_img/5--26.jpg' width=300 />
    <img src='test_output/5--26_Hosoda.jpg' width=300 />
</p>

<p>
    <img src='test_img/7--136.jpg' width=300 />
    <img src='test_output/7--136_Hayao.jpg' width=300 />
</p>

<p>
    <img src='test_img/15--324.jpg' width=300 />
    <img src='test_output/15--324_Hosoda.jpg' width=300 />
</p>

<p>
    <img src='test_img/6--267.jpg' width=300 />
    <img src='test_output/6--267_Hosoda.jpg' width=300 />
</p>

<p>
    <img src='test_img/4--24.jpg' width=300 />
    <img src='test_output/4--24_Hosoda.jpg' width=300 />
</p>


## Acknowledgement

- Many thanks to the authors for this cool work.

- Part of the codes are borrowed from [DCGAN](https://github.com/soumith/dcgan.torch), [AdaIN](https://github.com/xunhuang1995/AdaIN-style) and [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
