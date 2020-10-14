# Jasper-keras
## Jasper is an end-to-end convolutional neural acoustic model developed by Jason Li et al. (https://arxiv.org/pdf/1904.03288.pdf)

This is an implementation of the model in python and keras.


```
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from jasper_keras import keras_jasper


inputs = Input(shape=(100, 2))

output = keras_jasper(inputs, 4, 3, 1, Deep=True)

model = Model(inputs=inputs, outputs=output)

model.summary()
```
