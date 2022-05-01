# Project-group-19
The goal of Image Style Transfer is to take an input of two images, one for style and another for
content. The style image represents the texture, colors and curvatures, while the content image
represents the higher level objects and how they are arranged. The output is a combination of both
the style image and the content image.

## Setup
1. This project was made with torch==1.9.0 and torchvision==0.10.0+cu111 with Python 3.9.12
2. Git clone this repo, or put main.py into a folder where there is also an 'Output' folder and a 'data' folder. The output folder must contain three subfolders: 'random', 'cifar10', and 'cifar100', otherwise the code will not run.
4. Install the following dependencies

    _pip install torch torchvision_

## Run
To perfrom style transfer on cifar-10 dataset, run the following command

_python main.py cifar10 style-image-name_

To perfrom style transfer on cifar-100 dataset, run the following command

_python main.py cifar100 style-image-name_

To perfrom style transfer on random image, run the following command

_python main.py random input-image-name style-image-name_
  
All the output will be generated in output folder

To tune the style, add another parameter style-level and run the command

_python main.py random input-image-name style-image-name style-level 1_

_python main.py random input-image-name style-image-name style-level 2_

_python main.py random input-image-name style-image-name style-level 3_

Here, 1, 2 and 3 are three level of style where 3 is the highest intensity and 1 is the lowest.

## Results

<img src="Output/random/results.png" width = "1200" height = "650">
