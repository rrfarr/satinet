# Data Preparation

During this project we have used two datasets: i) the [Middlebury Stereo Vision](https://vision.middlebury.edu/stereo/)  that consists of stereo rectified images of synthetically generated objects 
and ii) the [IARPA Multi-View Stereo 3D Mapping Challenge Dataset](https://www.iarpa.gov/challenges/3dchallenge.html) that consists of fifty WorldView-3 panchromatic images collected between November 
2014 and January 2016 over San Fernando, Argentina.
The Middlebury Stereo Dataset was used to train the deep-learning models while the IARPA dataset was used for evaluation.
The following sections describe how to download and prepare both datasets.

## Middlebury Stereo Vision Dataset

The Middelbury Stereo Vision Dataset can be downloaded usimg the following script

```console
python data_preparation.py -d md
```
which will download and unpack all the required data files that will be used in the training of the models developed in this project.