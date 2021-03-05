# Training MC-CNN

The MC-CNN algorithm was trained on the Middelbury Stereo Vision Dataset. Please
make sure to download the dataset at half resolution using the command

```console
python data_preparation.py -d md
```

before proceeding. One can run train the MC-CNN network using the following command

```console
./mccnn-train.py -m Model/MC-CNN/ -d ./Data/MiddEval3/trainingH/
```

The attribute *m* specifies the folder where the model will be stored while the attribute
*d* specifies the directory containing the training data.
The list of training files is loaded from the md_list.json file.

