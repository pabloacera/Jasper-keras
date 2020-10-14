# Jasper-keras
## Jasper architecture developed by Jason Li et al. (https://arxiv.org/pdf/1904.03288.pdf)

This is an implementation of the model using python and keras.

Jasper BxR model has B blocks, each with R subblocks. 
Each sub-block applies the following operations: 
      1Dconvolution, 
      batch norm, 
      ReLU, 
      dropout.
All sub-blocks in a block have the same number of output channels.
Each block input is connected directly into the last subblock via 
a residual connection. The residual connection is first projected 
through a 1x1 convolution to account for different numbers of input 
and output channels, then through a batch norm layer. The output
of this batch norm layer is added to the output of the batch norm
layer in the last sub-block. The result of this sum is passed 
through the activation function and dropout to produce the output 
of the current block.
All Jasper models have four additional convolutional
blocks: one pre-processing and three post-processing.
https://arxiv.org/pdf/1904.03288.pdf

```
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from jasper_keras import keras_jasper


inputs = Input(shape=(100, 2))

# keras_jasper(inputs, R, B, output_untis, Deep=True)

output = keras_jasper(inputs, 4, 3, 1, Deep=True)

model = Model(inputs=inputs, outputs=output)

model.summary()
```
