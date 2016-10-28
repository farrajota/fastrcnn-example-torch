#Fast-RCNN code for torch7

[Fast-RCNN](https://github.com/rbgirshick/fast-rcnn) implementation for Torch7. This package allows to train, test and implement an object detector.


### Contents
1. [Features](#features)
1. [Requirements](#requirements)
3. [Usage](#usage)
3. [Fast RCNN package](#package)
4. [Training and testing](#Model train/test)
5. [Download dataset/pre-trained models](#Downloads)
5. [License](#License)
5. [Acknowledges](#acknowledges)

### Features
- Core functions wrapped in a Lua table for easier integration in a project
- Multi-threaded data load (uses torchnet as a backend)
- Multi-GPU support
- Support for additional data augmentation techniques (rotation, jitter, color shifts, etc.)
- Useful utilities like model creation, detection visualization, etc.


### Requirements

- Linux (not tested on MacOS)
- [Torch7](http://torch.ch/docs/getting-started.html)
- NVIDIA GPU with compute capability 3.5+ (2GB+ ram)
- At least 6GB free ram


### Installation

This Fast-RCNN package requires [Torch7](http://torch.ch/docs/getting-started.html) to be installed on your machine. Additionally, the following packages are required for this package to work:

```bash
luarocks install cudnn
luarocks install inn
luarocks install matio
luarocks install torchnet
(*optional*) git clone https://github.com/farrajota/dbcollection && cd dbcollection && luarocks make
```

The last package contains scripts to easily load/download datasets using a simple API. This package allows to load the necessary datasets for the example scripts. Also, as it is an optional package, it can ignored if you are only looking for core functionalities of the package.

### Running the demos

To run the basic demo code you'll first need to download/setup the pre-trained models. After this is done, just simply run the basic demo `examples/demo.lua` on the terminal:
```lua
qlua demo.lua
```

After running the demo you should see the following detections:

![alt text](data/demo/demo_detections.png "Detections with AlexNet")

To run a more dataset-oriented demo, instead run `examples/demo_db.lua` on the terminal while specifying the dataset of choice:

```lua
-- Pascal VOC 2007
qlua demo_db.lua -dataset pascal2007

-- Pascal VOC 2012
qlua demo_db.lua -dataset pascal2012

-- MSCOCO
qlua demo_db.lua -dataset mscoco
```

### Using this package

To use this package simply do 
```lua
local fastrcnn = require("fastrcnn")
```
This will load a table structure containing all the necessary functions to load/setup, train and test a FRCNN network. The core functions of this package provide scripts to [train](#train) and [test](#test) networks, to detect objects in images by using a [detector](#detector), and some [utility](#utils) functions useful to load and/or create FRCNN models, to load RoI proposals data from `.mat` files, and to visualize detection results with a window frame. 

<a name="train"></a>
#### train ####
```lua
fastrcnn.train(dataset, proposals, model, model_parameters)
```
<a name="test"></a>
#### test ####
```lua
fastrcnn.train(dataset, proposals, model, model_parameters)
```
<a name="detector"></a>
#### detector ####
```lua
local imdetector = fastrcnn.Detector(dataset, proposals, model, model_parameters)
```
<a name="utils"></a>
#### utils ####

```lua
local utils = fastrcnn.utils
```


### Training and testing a model using the example code

For training a Fast R-CNN network using the example code provided in `examples/`, you need to download the available pre-trained models, object proposals and the Pascal VOC 2007, 2012 and the MSCOCO datasets. This can be done by executing the following scripts: 
```lua
-- Download pre-trained models
th download_pretrained_models.lua

-- Download region proposals
th download_proposals.lua

-- Download+setup datasets
th download_datasets.lua
```
#### Train a network

Now you can train a model by calling ```th train.lua```. Also, you should specify the required options appropriately. For a list of complete options run ```th train.lua -help```. 

* You can select one of the following imagenet pre-trained networks for feature extraction: AlexNet, ZeilerNet, VGG (16, 19), ResNet (19, 34, 50, 101, 152, 200), and GoogleNet v3.
* The models together with text files describing the configuration, loss logs and model parameters of the training procedure will be saved into the specified path (default is `./data/exp`. You can change this directory by passing the `-expDir 'new path to save the experiment'` option. 

#### Test a network (mAP accuracy)

To test a previously trained network's mean average-precision, run the test script using `th test.lua` and define the `-expID`, `-dataset` and `-expDir` (if changed). 

Note: The mAP evaluation/testing script is always done using only a single GPU.


### TODO

. ~~ update CMakeList file ~~
. ~~ add rocks install file ~~
. update README
. add multi-gpu functionality to the roipooler
. augment roi prooposals by offsetting its positions around the original box
. test voc 2012 and mscoco datasets
. add nms.c from multipathnet code for faster evaluation

### License

The present code is released under the BSD License (refer to the LICENSE file for more details).

### Acknowledges

This work was developed and inspired by the following repositories: [Fast-RCNN*](https://github.com/rbgirshick/fast-rcnn), [Fast-RCNN for Torch7*](https://github.com/mahyarnajibi/fast-rcnn-torch) and [facebook/multipathnet*](https://github.com/facebookresearch/multipathnet). 

(*all rights reserved to the corresponding authors)
