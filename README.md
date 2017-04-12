# Project: Traffic Sign Classifier
### Overview
This project uses a ConvNet architecture to classify traffic signs. The dataset to train are traffic sign imgaes from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

##### For more information about this project visit the [Wiki page](https://github.com/cuevas1208/Traffic_Sign_Classifier/wiki)

### Files Overview
                                                                                                                          
`model_architecture.py`

Model architecture

`model_calls.py`

Functions to train and use the model

`preprocess_augmentation.py`

Split, balance and augmentation functions

`helper_Functions.py`

Functions to visualize data

### Dependencies
This project requires **Python 3.5** and the following Python libraries installed:
- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`

### Dataset 
Data set for this project can be downloaded from [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 
NOTE: Data set images used on this project have been already resized to 32x32. You may want to do that if you you are using different image size. 

### Quickstart
Clone the project and start the notebook.
```
git clone https://github.com/cuevas1208/Traffic-Sign-Recognition
cd Traffic-Sign-Recognition
python main.py
```

### Additional sources
This project is part of Udacity Self-Driving Car Engineer Nanodegree program. Tools, techniques and knowledge learned in class about deep neural networks and convolutional neural networks were used to classify traffic signs.

To learn more about convolutional networks I recommend this [book](http://www.deeplearningbook.org/contents/convnets.html)
