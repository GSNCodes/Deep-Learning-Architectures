## ResNet Architecture

<img src = 'misc/resnet-architectures-34-101.png'>  

The architecture of all the variants of ResNet is shown above.  


### ResNet Block
Here is a sample resnet block that shows the use of skip-connections.  
<img src = 'misc/resnet-block.png'>  

## Notes
You can find my implementation of all the variants of ResNet in `resnet.py`.
The variants are:-
- ResNet-18
- ResNet-34
- ResNet-50
- ResNet-101
- ResNet-152  

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

## Example Commands

You can import the resnet module as given below  
`from resnet import resnet18, resnet34, resnet50, resnet101, resnet152`  

For ResNet-18:  
`model = resnet18(num_classes=..)`  
  
For ResNet-34:  
`model = resnet34()`  
  
For ResNet-50:  
`model = resnet50()`  
  
For ResNet-101:  
`model = resnet101()`  
  
For ResNet-152:  
`model = resnet152()`  
  
  
## Reference
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
