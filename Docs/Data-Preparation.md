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

## IARPA Multi-View Stereo 3D Mapping Challenge Dataset

The IARPA Multi-View Stereo 3D Mapping Challenge Dataset is hosted by SpaceNET on Amazon Web Services (AWS).
This can be downloaded using 

```console
aws s3 ls s3://spacenet-dataset/mvs_dataset 
```
More information can be found on [SpaceNet](https://spacenetchallenge.github.io/datasets/mvs_summary.html).
The [AWS Command Line Interface (CLI)](https://aws.amazon.com/cli/) must be installed with an active AWS account. Configure the AWS CLI using ‘aws configure’

This repository will provide access to the following data
1.  Updated metric analysis software with examples from contest winners 
2.  Challenge data package with instructions, cropped TIFF images, ground truth, image cropping software, and metric scoring software (1.2 GB) 
3.  JHU/APL example MVS solution (451 MB) 
4.  NITF panchromatic, multispectral, and short-wave infrared DigitalGlobe WorldView-3 satellite images (72.1 GB)
5.  LAZ lidar point clouds with SBET (2.2 GB)
6.  Spectral image calibration software (84 MB)

Alternatively, the GEOTIFF files can be downloaded using the following script

```console
python data_preparation.py -d iarpa -l 18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.TIF 18DEC15WV031000015DEC18140544-P1BS-500515572060_01_P001_________AAE_0AAAAABPABJ0.TIF```
```

where 18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.TIF and 18DEC15WV031000015DEC18140544-P1BS-500515572060_01_P001_________AAE_0AAAAABPABJ0.TIF
are the GEOTIFF filenames that will be downloaded from a secondary website.



