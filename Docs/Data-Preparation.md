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
To do this first [create and activate a AWS account](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/?nc1=h_ls).
The use IAM to [create a user with the access keys](https://aws.amazon.com/iam/faqs/). You have to use 

```console
aws configure
```
to configure AWS using the Access key ID and Secret access key provided. If missing, use **us-east-1** as default region name and **json**
as default output format. You can then check the information about the dataset using

```console
aws s3 ls s3://spacenet-dataset/Hosted-Datasets/MVS_dataset/
```

The dataset can then be downloaded using
```console
aws s3 cp s3://spacenet-dataset/Hosted-Datasets/MVS_dataset/
```

More information can be found on [SpaceNet](https://spacenetchallenge.github.io/datasets/mvs_summary.html).
The [AWS Command Line Interface (CLI)](https://aws.amazon.com/cli/) must be installed with an active AWS account. Configure the AWS CLI using ‘aws configure’

Alternatively, the GEOTIFF files can be downloaded using the following script

```console
python data_preparation.py -d iarpa -l 18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.TIF 18DEC15WV031000015DEC18140544-P1BS-500515572060_01_P001_________AAE_0AAAAABPABJ0.TIF```
```

where 18DEC15WV031000015DEC18140522-P1BS-500515572020_01_P001_________AAE_0AAAAABPABJ0.TIF and 18DEC15WV031000015DEC18140544-P1BS-500515572060_01_P001_________AAE_0AAAAABPABJ0.TIF
are the GEOTIFF filenames that will be downloaded from a secondary website.



