# The SATINET project
The SATINET project presents a deep learning based stereo-vision algorithm that was integrated within the Stereo Pipeline for Pushbroom Images (S2P) framework. The proposed stereo matching 
method applies a siamese convolutional neural network (CNN) to construct a cost volume. A median filter is applied to every slice in the cost volume to enforce spatial smoothness. We then 
apply another CNN to estimate the confidence map which is then used to derive the final disparity map. Simulation results on the IARPA dataset have shown that the proposed method achieves a 
gain of 4.5% in terms of completeness. A qualitative assessment reveals that the proposed method manages to generate DEMs with less noise. The proposed method adopts the architectured
depicted in the figure below.

![Diagram of the proposed method](./Figures/diagram.png)*Diagram of the stereo matching method proposed by the SATINET project*

In order to assess the performance of the proposed pipeline, we selected a subset of the well-known IARPA challenge dataset. We use two WorldView-3 images acquired on 18 December 2015 on 
the same track. The proposed stereo matching process was integrated within the S2P pipeline which, like most stereoscopic DEM methods, normally uses SGBM for stereo matching. 
The table below compares our approach against the standard S2P pipeline. Our first contribution is the S2P-MCCNN method, which uses MC-CNN for stereo matching, and outperforms 
S2P-SGBM by 3% in terms of completeness. The addition of a median filter to smoothen the cost volume slices (S2PMCCNN-Filt) provides an additional gain of 1.3%. Finally, using LAF-Net to fuse the left and right disparities (S2PMCCNN-
Filt-LAFNet) results in the best performance, outperforming S2P-SGBM by 4.5%.

| Syntax      | Description |
| ----------- | ----------- |
| Header      | Title       |
| Paragraph   | Text        |

# Installation

To execute the software you need to install the virtual environment. This can be easily
done using the following command

```console
./install.sh
```
This will install all the packages within the requirements.txt file.
In case the virtual environment is not activated, you can do so by running the following command

```console
source venv/bin/activate
```
and can be deactivated using

```console
deactivate
```

This code was tested on a Ubuntu 18.04 operating system using Python 3.6, Tensorflow 1.14.0 and pytorch 1.8.0.
A more complete list of packages used can be found in the install.sh script.

# Documentation
1.  [Data Preparation](./Docs/Data-Preparation.md)
2.  [Training the MC-CNN Network](./Docs/Training-MCCNN.md)
3.  [Stereo Matching](./Docs/Stereo-Matching.md)
4.  [Evaluation](./Docs/Evaluation.md)