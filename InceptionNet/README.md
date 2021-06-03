## InceptionNet - GoogLeNet Architecture

<img src = 'misc/inception_module.png'>

The architecture details of GoogLeNet is tabulated down below:-  

<img src = 'misc/GoogLeNet_Architecture.png'>

## Notes

You can find the model implementation in `InceptionNet.py` file.  
The training code can be found in `train.py`.  
  
I have ensured that maximum control is given to the user.  
* You can set whether you want to include Batch Normalization layers in the Convolution Block by setting the `use_batchnorm` parameter to `True` or `False`.
* Similarly, you can choose whether to include the auxiliary classifier when training the model by setting the `aux_network` parameter to `True` or `False`.

For example,  
`model = GoogLeNet(in_channels = 3, num_classes = 10, aux_network = True, use_batchnorm = True)`

This makes sure that the model includes both Batch Normalization layers as well as the Auxiliary Classifiers during training.

The directory structure for the dataset is supposed to be like this:-
```
dataset/
  ----train_set/
    ----Class-1/
    ----Class-2/
    .
    .
    .
    ----Class-n/
    
  ----test_set/
    ----Class-1/
    ----Class-2/
    .
    .
    .
    ----Class-n/
```

Each "class folder" should contain it's corresponding images. 
By default the number of classes is set to 10. In the `train.py` script the classes would be set to 2 when the model was created. Feel free to change this according to the dataset you use.

## References
[Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)


