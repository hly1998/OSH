# OSH
The codes for a paper

# Main Dependencies
+ pytohn 3.8
+ torch 1.11.0+cu113
+ numpy 1.22.4
+ psutil 5.9.1
+ kornia 0.7.1
+ pandas 2.0.3

# How to run
You can easily run our code by following these steps: 

+ Replace "{your root}" in the file "utils/tools.py" with your own file path.
+ In OSH directory, run the command "sh scripts/main.sh" to begin the training process.

You will obtain the results of our model with different BNN backbones on the CIFAR100 dataset. Please note that the CIFAR100 dataset will be automatically downloaded when you run the code.
