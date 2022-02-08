# Image Captioning

## Project Description

For the purposes of website accessibility, images often need to have corresponding alternative text in order for screen readers to convey that type of content to its users. Currently, much of this alternative text is being generated manually, which is very time-consuming and potentially never-ending work. 

Since the images vary greatly in size, resolution, and content, it is hard to easily generate accurate and descriptive alternative text for them. Recent solutions using advanced computer vision have difficulties with the complex semantics that some of the images have. 
We aim to train a neural network that can overcome these existing challenges. Using a Kaggle data set that comes from Wikipedia, we will create an English-based training data set for the neural network. Since the training data is about 275GB, we will have testing data that is cleaned and filtered to be able to produce the best results.

In terms of ethics, we should be doing this in order to increase accessibility for web users who require a screen reader to interpret pages. The accuracy of a non-ML alternative could be very high but would require significantly more human work. We will work to minimize bias in our data by providing diverse sets of images and seek to explore how current models misinterpret results and how these can be prevented. Consequently, we will research how individuals' privacy plays into the collection of image recognition data such as facial recognition and how models are able to protect people’s anonymity. 

Ultimately, our long-term goal is to use the neural network to caption or label any given image.

## Project Goals

1. Create data sets for training and testing neural networks that creates captions for images
2. Explore methods for cleaning the training and test data sets
3. Train a neural network that is able to caption any image
