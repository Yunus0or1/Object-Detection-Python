# Object-Detection-Python
This repo contains projects on implementaion of different object detection algorithm.

# Installation

  - Install python 3.6.2 64-bit version or Newest Version
  - Run these commands : 
    ```
    pip3 install tensorflow==1.14
    pip3 install tensorflow-gpu==1.14 (Stable but visual C++ 2015 v3 update required)
    ```
 - Install cuda 10 
 - Download cudNN from Nvidia after Login.
 - Copy contents of cuDNN 10 to C:\Program Files\NVIDIA GPU Computing Toolkit. I have used cudNN v11.
 - You might need other cudNN downloads to copy-paste dll files.

>You may find several missing dll. Just find them on internet or go to C:\Program Files\NVIDIA GPU Computing Toolkit. Find the similar dll and rename it.

**This is the chart to make accurate versioning between tensorflow GPU, cuda and cudNN**

>[StackOverFlow](https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible)
>[TensorFlow](https://www.tensorflow.org/install/source#tested_build_configurations)


# Check TensorFlow

To check if your GPU is enlisted:

```Python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 0 = GPU use; -1 = CPU use
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
```
[Medium](https://medium.com/@liyin2015/tensorflow-cpus-and-gpus-configuration-9c223436d4ef)







