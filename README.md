# The SATINET project
The SATINET project presents a deep learning based stereo-vision algorithm that was integrated within the Stereo Pipeline for Pushbroom Images (S2P) framework. The proposed stereo matching 
method applies a siamese convolutional neural network (CNN) to construct a cost volume. A median filter is applied to every slice in the cost volume to enforce spatial smoothness. We then 
apply another CNN to estimate the confidence map which is then used to derive the final disparity map. Simulation results on the IARPA dataset have shown that the proposed method achieves a 
gain of 4.5% in terms of completeness. A qualitative assessment reveals that the proposed method manages to generate DEMs with less noise. The proposed method adopts the architectured
depicted in the figure below.

![Diagram of the proposed method](./Figures/diagram.png)*Diagram of the stereo matching method proposed by the SATINET project*

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
# Documentation
1.  [Data Preparation](./Docs/Data-Preparation.md)
2.  [Training the MC-CNN Network](./Docs/Training-MCCNN.md)
3.  [Stereo Matching](./Docs/Stereo-Matching.md)
4.  [Evaluation](./Docs/Evaluation.md)