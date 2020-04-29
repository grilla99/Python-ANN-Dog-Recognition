# Abstract

The aim of this task was to explore whether an MLP could learn to perform correct classification of dog breeds on an unseen image by training the network using self-collected datasets. Successful identification of these dogs was achieved using Python and various deep learning suites and computer vision tools including Keras and Tensorflow. The most popular tasks for performing image classification typically use the CIFAR-10 image dataset and the MNIST handwritten image dataset and thus I was eager to explore whether a relatively ‘shallow’ network could be taught a challenging task such as dog breed identification. A potential real-life implementation of this software includes use in animal rescue where poorly kept dogs may be difficult to identify, and this software can give an insight into the breed. This task was a great success and solidified my understanding of Neural Networks and gave myself a great introduction to Python.


# Introduction

Throughout the course of the Autumn term we as a class was introduced to the concept of artificial neural networks, activation functions and multi-layer perceptron networks. Such work saw development of single layer networks using Java in order to solve linearly separable problems (such as OR) and subsequently the development of an MLP network in order to solve non-linearly separable problems i.e. exclusive-OR. Developing this provided a good foundational understanding of neural networks and also implementing object-oriented techniques in Java.

After this we was given a chance to attempt to use our networks to solve an arbitrary issue using our network. I got fascinated with the concept of wanting to perform image classification however, after many attempts using my Java network and a lot of research concluded that this would be better implemented in Python [1][2] despite my lack of existing knowledge in the language. As previously mentioned, there is lots of examples and lots of tutorials on implementing image recognition using the MNIST and CIFAR-10 datasets which made me eager to differentiate and try to use a different dataset. I possess a love for dogs and thus decided to implement a network that would be trained to identify dog breeds.

# Implementation
## Data Gathering

  Resources available online yielded many possible datasets related to the image recognition however often these were archives ranging from 0 – 100 gigabyte data which was simply too large a dataset using a local machine to train a network and would have spent too much time training without allowing for errors. I did stumble across an existing dataset [3] [4] which had been created via combining two datasets used for computer vision tasks by researchers at Stanford university. The issues with this data is that it included 20000 images and 120 breeds which again I believed would be out of the scope of the networks feasible capabilities in terms of ability to process and thus I continued my search for something more appropriate.
  
  Eventually, I discovered an optimal tool developed by Microsoft / Bing within their cognitive services packages called the Bing Image Search API which is conveniently compatible with Python. [5][6] This tool essentially allows for a Bing image search to be performed and the subsequent results of the query to be output to a folder of the users’ choice. To make it a bit more interesting than binary classification I chose to have three classes of which these were a Cocker Spaniel, a German Shepherd and a Labrador retriever.
  
  To gather the data I ran the python script that I had created that interacts with the BingAPI and provided parameters of the dog name and also the number of results to be returned. I wanted to have a reasonable amount of data to train with but not so large it would be negligible to the training time and thus I decided 450 images for each class should suffice and directed these to a ‘dataset’ folder. The next stage was rather tedious and involved manually checking all 1350 images for any irrelevant images or corrupted files and remove them so as not to affect the training of the network.
  
  After removing this extraneous data, I was left with 423 images of Cocker Spaniels, 395 of German Shephard and 412 images of Labradors. These included action shots and still shots, these were all included in their own directory named Cocker, German and Labrador respectively.

## Data Processing

  In order to create the neural network in python you must specify the filters and the size of the kernel. My layer had 32 filters with a 3 x 3 kernel as such the spatial dimensions of the input images (which was selected to be 96 x 96 x 3 to train the network) can be reduced from 96 x 96 to 32 x 32 and allow for faster training. Thus, each image is re-sized prior to processing.
  
  In addition to this the following pre-processing was also required. The ImageDataGenerator [7] class from the high-level neural networks API Keras was used to apply random transformations to the dataset to create additional training data, this is used also to prevent overfitting in the network. [8]
Figure 1. Data augmentation code
  
  A common topic in the field of machine learning is how categorical data can be encoded into integers such that ML algorithms are able to perform better. [9] The LabelBinarizer tool within sci-kit python library allows class labels (Cocker, German, Labrador) to be transformed into one-hot encoded vectors. [10] Thus allowing me to work with human readable labels on the surface and for the data to be converted into integer format under the hood and fed to the network.
  
  Finally, for the data to be passed to the network the pixel intensities (of each pixel in the image) must be scaled to the range [0,1] as using the unsigned integers (range [0,255]) can result in slower changes during modelling.[11]

## Training the network

  With the data successfully pre-processed it was time to train the network. I had seen on some tutorials provisioned by Google [12] and others that training : test data split is typically between 70 : 30 and 80 : 20. At first I used a training split of 80:20 however this resulted in quite severe overfitting. After some toying around I found settled on a 75:25 split on the data and this yielded the results seen in Figure 2. The data split was performed randomly on the images using train_test_split module from scikitlearn. The training was executing on an anaconda environment in order for the necessary packages to be imported correctly as there was compatibility issues between Tensorflow and Python 3.7 causing program errors. On my desktop the training time took about 60 – 90 minutes and on my laptop, it took 120 – 160 minutes.
  
  As can be seen from the data the higher validation loss compared to the training loss suggest some overfitting and this becomes alarming at around 50 epochs, however this value falls dramatically following this and settles down to a value not much greater than the training loss at 100 epochs, so whilst there is still some overfitting it is not negligible to the network. I believe this to be caused by some superstitions learned in the training data that don’t have a true basis in reality and thus causing the difference to the validation data.

# Results

  Shown in Figure 3 are the results from the training of the network. At epoch 100 it managed to achieve 90.3% classification accuracy on the training set and 79.10% accuracy on the training data, highlighting the aforementioned overfitting.
Figure 3. Output from training network
  
  Having achieved a decent level of accuracy on my training and test data I was then eager to test my neural network. In general, it can be said that the network was able to correctly classify arbitrary images aside from a few anomalies. Beginning with the most interesting results first, I tested my networks ability to classify dogs that weren’t ‘real’ dogs. As can be seen in Figure 4 and 5 the network was able to correctly classify an image of a painting of a Cocker and a plush toy Labrador demonstrating the extent the network can extrapolate to alternate cases.
  
  In order to rule out the possibility that the test images the network had been successfully identifying by an extraneous variable I then proceeded to use images of friends’ dogs that couldn’t have been anywhere online and put the network to the test. Shown in Figure 6 and 7 is the network correctly identifying the breeds of the dogs.
  
  However, when using an image of my own pure breed cocker spaniel the network incorrectly identified this as a Labrador with a high degree of certainty which as can be seen in Figure 8 is incorrect. In future it would be preferable to allow the network to learn dynamically from incorrectly identifying on the testing data.

# Discussions
## Discussion of results

  The results obtained allowed me to notice some interesting patterns amongst the data. Firstly, is that the network is better at identifying images that have a plain background and the dog is the main focus. It had trouble identifying side profiles or action shots however I suspect that is due to lack of training data and will be remedied with a larger volume of training images.
  
  Furthermore, it was also interesting to see that whilst the network was very good at classifying Labradors and German Shepherds it was, in comparison, incompetent at identifying the Cocker consistently. As seen in the previous example with my dog and also in Figure 9 it frequently mistook a cocker spaniel for a Labrador. I am yet to uncover a reason for this however my theory is that that it may be due to the lack of training data causing the network to extract higher level features from the cocker images that individually point towards a Labrador more however when combined form a cocker.
  
  In addition to this the success of the network was displayed when identifying friends’ dogs and this was pleasing to see as it showed that the network wasn’t just good at identifying online images from a dataset and could operate using ‘real life’ images. It frequently identified the correct dog with a high % degree of confidence.
  
 ## Final Discussion
 
 This task was a very interesting task and allowed me to learn so much knowledge in such a short space of time. My knowledge of neural networks at the beginning was shaky and my knowledge of python was non-existent so actually creating the network was very challenging and took many hours but I have come out of it with a very good understanding of python, virtual environments, terminal environments, python libraries and has made me confident in my understanding of neural networks. I think that I made it more difficult by choosing the task I did but I am pleased with the system produced. It significantly improved my time management skills and resilience through the course of development.

