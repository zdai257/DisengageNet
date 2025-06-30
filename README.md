# Gaze Target 360

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![Demo1](gifs/demo1.gif)

![Demo1](gifs/demo2.gif)

## Description

This is the software system of GazeTarget360. The aim is to estimate gaze target anywhere, incl. in-frame target (IFT), out-of-frame target (OFT), and eye-contact (EC), from a generalized visual scene. GazeTarget360 integrates conditional inference of a ResNet eye-contact [detector](https://github.com/rehg-lab/eye-contact-cnn) and a transformer encoder-decoder based gaze location estimator. GazeTarget360 makes a first-of-its-kind system to predict gaze targets from realistic camera footage for robotic perception.

## Prerequisites

The following libraries are required

- Python = 3.11
- pytorch = 2.5.1
- torchvision = 0.20.1
- dlib (OR transformers)
- wandb

This work is inspired by [Gaze-LLE](https://github.com/fkryan/gazelle/tree/main?tab=readme-ov-file) model. You can create their conda environment provided then install *dlib* library. You will need to download [VideoAttentionTarget](https://github.com/ejcgt/attention-target-detection) dataset in /VAT and [GazeFollow](https://www.dropbox.com/scl/fi/n45q7wig1rvrqf8hsomuw/gazefollow_extended.zip?rlkey=e5b54qgppse4xfk4wc6j2zj2f&e=1&dl=0) dataset in /gazefollow_extended under the root folder for training.

## Steps

Run *preprocess_X.py* to prepare formatted annotations of datasets. Use the following command to pre-train the model on GazeFollow

```base
python train_gazefollow.py
```

then use the following command to fine-tune the model on VideoAttentionTarget

```base
python train_videoattentiontarget.py
```

For ColumbiaGaze, EYEDIAP, and MPIIFaceGaze, download the datasets and use *Eval_X.py* to evaluate performance. 

To infer gaze targets, run the following command to obtain eye-contact, in-frame, or out-of-frame estimation results with visualization

```base
python Demo_sys.py
```

OR run *Demo_detr.py* to use a DETr-based face detection front end.

Trained [models](https://drive.google.com/drive/folders/1_JR-gMJtT1pV00BsO_0Q1Y6PA1qNzSkq?usp=sharing) of GT360 are released.

![openingfig](https://github.com/zdai257/DisengageNet/blob/main/processed/demo0.jpg)

If you use our work, please cite:

```bibtex
@INPROCEEDINGS{gazetarget360_iros2025,
  author    = {Dai, Zhuangzhuang and Zakka, Vincent Gbouna and Manso, Luis J. and Li, Chen},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title     = {GazeTarget360: Towards Gaze Target Estimation in 360-Degree for Robot Perception}, 
  year      = {2025}
}
```

## Acknowledgements

Experiments were run on Aston Engineering and Physical Science Machine Learning Server, funded by the EPSRC Core Equipment Fund, Grant EP/V036106/1.

