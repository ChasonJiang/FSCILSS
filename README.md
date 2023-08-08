# Few-shot Class-Incremental Semantic Segmentation via Pseudo-Labeling and Knowledge Distillation

<!-- ![overview](./imgs/overview.svg) -->
<p align="center">
  <img src="./imgs/overview.svg" width="500"/>
</p>


Link to our Paper: https://arxiv.org/abs/2308.02790


## Prerequisites and Requirements
We tested our code under Ubuntu 20.04 and Windows with

    - Python 3.11.4
    - Cuda 11.8
    - PyTorch 2.0.1

To run our code please proceed as follows:
1. Install all packages from the ```requierements.txt``` in the root folder. 
2. You must specify the absolute path of the Dataset and Checkpoints directories in the Path.json file of the project root directory
3. Download the [Cityscapes dataset](https://www.cityscapes-dataset.com/) and [KITTI dataset](https://www.cvlibs.net/datasets/kitti/),then put it in a folder "Dataset".For the Cityscapes dataset, you need to download some [JSON files](https://drive.google.com/file/d/1Y423qcJnYA7QktE73fneNDB6ZR7oBhzA/view?usp=sharing),and extract them before placing them in the folder "cityscapes".
4. Your Dataset folder should look like this:
```
├───cityscapes
│   ├───gtFine
│   │   ├───test
│   │   ├───train
│   │   └───val
│   └───leftImg8biWt
│       ├───test
│       ├───train
│       └───val
└───kitti
    ├───test
    └───train
```

## Evaluation
You can evaluate the trained models with the following command:
```
bash ./evaluate.sh
```

## Training

The following commands can be used to train the model:
```
bash ./train.sh
```

## Pre-trained models
You can download our pre-trained models [here](https://drive.google.com/file/d/1CcsAx7TZr6me3lTe0hoQIpDZr68LvwPW/view?usp=sharing),and unzip to the project root directory.


## License

The <b>ERFNet</b> model used in this project was developed by E. Romera et al. <a href="https://github.com/Eromera/erfnet_pytorch">here</a>. The project was released under the *Creative Commons Attribution-NonCommercial 4.0 International License*. Our Code is also licensed under the *Creative Commons Attribution-NonCommercial 4.0 International License*, which allows for <strong>personal and research use only</strong>.

View the license summary here: http://creativecommons.org/licenses/by-nc/4.0/
