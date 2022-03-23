# CPSN-OSFG.PyTorch

[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.1.0-%237732a8)

A simple PyTorch implementation of CPSN (Coupled Patch Similarity Network) for OSFG (One-Shot Fine-Grained Image Recognition). 

[Coupled Patch Similarity Network For One-Shot Fine-Grained Image Recognition](https://cser-tang-hao.github.io/Publications/ICIP2021-Coupled%20Patch%20Similarity%20Network%20For%20One-Shot%20Fine-Grained%20Image%20Recognition.pdf), ICIP2021

## Environment
This code is implemented on PyTorch and the experiments were done on a NVIDIA TITAN RTX GPU.
So you may need to install
* Python==3.x
* torch==1.1.0 or above
* torchvision==0.3.0
* tqdm

## Datasets
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). <br>
- [StanfordCar](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). <br>
In our paper, we just used the CUB-200-2011 dataset and StanfordCar dataset. Note, if you use these datasets, please cite the corresponding papers. 


##  Usage
This repo contains CPSN with feature extractors using Conv64 / ResNet12 in PyTorch form, see ```./models/```. 

* resnet: Conv64|ResNet12.
* nExemplars: number of training examples per novel category.
* nKnovel: number of novel categories.
* gpu_devices: gpu id.
* scale_cls: scale the metric scores.
* phase: use test or val dataset to early stop.
* num_classes: number of all training categories.

### Train a 5-way 1-shot model based on Conv64 or ResNet12:

```shell
python ./train.py --dataset CUB  --resnet 0 --num_classes 100 --nExemplars 1
python ./train.py --dataset CUB  --resnet 1 --num_classes 100 --nExemplars 1
```

### Test the model:

```shell
python ./test.py --dataset CUB  --resnet 0 --num_classes 100 --nExemplars 1
python ./test.py --dataset CUB  --resnet 1 --num_classes 100 --nExemplars 1
```

## Citation
If this work is useful in your research, please cite 

```
@inproceedings{tian2021coupled,
  title={Coupled Patch Similarity Network FOR One-Shot Fine-Grained Image Recognition},
  author={Tian, Sheng and Tang, Hao and Dai, Longquan},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={2478--2482},
  year={2021},
  organization={IEEE}
}
```

## References
This implementation builds upon several open-source codes. Specifically, we have modified and integrated the following codes into this repository:

*  [CAN](https://github.com/blue-blue272/fewshot-CAN) 

