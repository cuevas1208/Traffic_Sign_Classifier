# Project: Traffic Sign Classifier
This project is part of Udacity Self-Driving Car Engineer Nanodegree program. Tools, techniques and knowledge learned in class about deep neural networks and convolutional neural networks were used to classify traffic signs.

### Overview
In this project, ConvNet architecture is used instead of a simple Feed Forward Network to classify traffic signs. The dataset to train are traffic sign imgaes from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

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

### Get started
Clone the project and start the notebook.
```
git clone https://github.com/cuevas1208/Traffic-Sign-Recognition
cd Traffic-Sign-Recognition
jupyter notebook Traffic_Signs_Recognition.ipynb
```

### To Improve or Issues
Visualizing the model architecture using TensorBoard.

Instead of a fixed number of epochs, implemente an early termination, as overtraining can lead to overfitting. 

Rebalance the number of examples for each class. This has the potential to improve your results here.

### Additional sources
To learn more about convolutional networks I recommend this [book](http://www.deeplearningbook.org/contents/convnets.html)
