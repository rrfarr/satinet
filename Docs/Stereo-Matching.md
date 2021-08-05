# Stereo Matching

This script will use the [S2P processing pipeline](https://github.com/MISS3D/s2p) developed by CNES which rectifies, matches and derives the 
Digital Elevation Model (DEM) from two or more satellite images.
One can run the matching script using the following command

```console
./matching.py  --in_foldername ./Data/Iarpa --method s2p-mccnn --mccnn_model_path ./Model/MC-CNN/
```

This script takes three command line inputs. The *in_foldername* argument specifies the foldername where the GEOTIFF images are stored.
The *method* argument specifies the stereo-matching method used.
The SATINET library supports three stereo matching methods: i) *s2p* which employs the [SGBM algorithm](https://ieeexplore.ieee.org/document/4359315) (default) 
ii) *s2p-mccnn* which employs the MC-CNN algorithm for stereo-matching together with a Median Filter to each cost-volume slice and iii) *s2p-mccnn-laf* which
applies the stereo matching described in  *s2p-mccnn* but also fuses the left and right disparity using the LAFNET network.
Finally, the *mccnn_model_path* is the path where the checkpoint folder of the model that was previously trained using the mccnn-train.py script is.