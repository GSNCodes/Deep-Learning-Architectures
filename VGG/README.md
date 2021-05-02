## VGG Architecture

![](vgg.png)

## Notes

The `VGG_Net.py` script contains the VGG16 and VGG19 models.

`main.py` contains code to train the models in the above file.  
You can choose between the two models as follows:-  
`python main.py --model vgg16` or `python main.py --model vgg19`

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
By default the number of classes is set to 10. In the `main.py` script the classes would be set to 2 when the model was created. Feel free to change this according to the dataset you use.

## References
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
