# Gaze Target 360

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![Demo1](gifs/demo1.gif)

![Demo1](gifs/demo2.gif)

## Description

This is the software system of GazeTarget360. The aim is to estimate gaze target in 360-degree from a generalized visual scene. GazeTarget360 integrates conditional inference of a ResNet eye-contact detector and an encoder-decoder based in-frame target and out-of-frame target estimator. GazeTarget360 makes a first-of-its-kind system to predict gaze targets from realistic camera footage for robotic perception.

## Prerequisites

The following libraries are required

- Python = 3.11
- pytorch = 2.5.1
- torchvision = 0.20.1
- dlib

This work is inspired by [Gaze-LLE](https://github.com/fkryan/gazelle/tree/main?tab=readme-ov-file) model. You can create their conda environment provided then install *dlib* library. You will need to download [VideoAttentionTarget](https://github.com/ejcgt/attention-target-detection) dataset in /VAT and [GazeFollow](https://huggingface.co/datasets/ShijianDeng/gazefollow) dataset in /GazeFollow under the root folder for training.

## Steps

Use the following command to pre-train the model on GazeFollow

```base
python train_gazefollow.py
```

then use the following command to fine-tune the model on VideoAttentionTarget

```base
python train_vat.py
```

To evaluate the results on Columbia, EYEDIAP, and MPIIFaceGaze, download the datasets and use *preprocess_x.py* to preprocess the raw data. Use *Eval_x.py* to evaluate the performance. 

To infer gaze targets, run the following command to visualise eye-contact, in-frame, or out-of-frame estimation results with visualization

```base
python Demo_sys.py
```

![openingfig](https://github.com/zdai257/DisengageNet/blob/main/processed/demo0.jpg)

## Acknowledgements

Experiments were run on Aston Engineering and Physical Science Machine Learning Server, funded by the EPSRC Core Equipment Fund, Grant EP/V036106/1.

