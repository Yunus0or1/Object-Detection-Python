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
[Tutorial from Medium](https://medium.com/@liyin2015/tensorflow-cpus-and-gpus-configuration-9c223436d4ef)


# Check Keras

```Python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 0 = GPU use; -1 = CPU use

import keras
import tensorflow as tf

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 3} )
sess = tf.compat.v1.Session(config=config)
keras.backend.set_session(sess)
```

>If you face problem with 'import keras.something' convert it to'tensorflow.python.keras.something'


# TensorFlow Version check 

```Python
import tensorflow as tf
print(tf.version.VERSION)
```

# MaskRCNN installation

Download MatterPort Github repo on MaskRCNN
Use it according to your need. Read README.MD

[Tutorial Link from YouTube](https://www.youtube.com/watch?v=GSDbfGsxruA&t=174s)


# PycocoTools

>Most cruicial. No good update for windows 10. [Visit here](https://github.com/philferriere/cocoapi)

# Resources

>[Yolov3](https://github.com/ultralytics) |
>[Yolov3 Tutorial with direct weight](https://www.youtube.com/channel/UCtfTf1nNJQ4PbUDqj-Q48rw) |
>[Convert to Coco Format](https://medium.com/datadriveninvestor/how-to-create-custom-coco-data-set-for-instance-segmentation-68dbfc988b56)


___
**Project Explanation**
___

- A very details guideline on Mask-RCNN is published in [Medium](https://medium.com/@ahmedyunuspilot/mask-rcnn-on-custom-coco-like-dataset-on-windows-machine-7e219b6a1fc3) by the Author **Ahmed Yunus**.
- However to understand YOLO, this [link](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/) is referred. 
- The codes are self explanatory. The function names are completely aligned with what those are meant to do. 


