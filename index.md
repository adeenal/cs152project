# Image Captioning

## Team Members
+ Adeena Liang
+ Irmak Bukey
+ Olina Wong
+ Sadie Zhao

## Introduction
For the purposes of website accessibility, images often need to have corresponding alternative text in order for screen readers to convey content in the images to its users. Currently, much of this alternative text is being generated manually, which is very time-consuming and potentially never-ending work. Often known as “automatic image annotation” or “image tagging” the problem of generating textual descriptions for images presents itself as a complex problem for models.

Since the images vary greatly in size, resolution, and content, it is hard to easily generate accurate and descriptive alternative text for them. Recent solutions using advanced computer vision have difficulties with the complex semantics that some of the images have. Image descriptions with text have historically fallen into three variations:

1. Classification - Where the model assigns an image a class label from known classes
2. Description - Where the model generates a textual description of the overall image contents
3. Annotation - Where the model generates descriptions specifically for certain regions on a given image

For this project, we aim to train a neural network falling into the second variation that can overcome these existing challenges of variations and semantics. Using a Kaggle data set that comes from Wikipedia, we will create an English-based training data set for the neural network. Since the training data is about 275GB, we will have testing data that is cleaned and filtered to be able to produce the best results.

## Project Goals

1. Create data sets for training and testing neural networks that creates captions for images
2. Explore methods for cleaning the training and test data sets
3. Train a neural network that is able to caption any image

# Literature Review
During our research process, we examined both research articles and existing implementations which relate to our project topic. Similarly, we also investigated academic literature and relevant blog postings to gain stronger understandings of how to implement our project both successfully and effectively.

In a [blog posting](https://machinelearningmastery.com/how-to-caption-photos-with-deep-learning/) focused around machine learning, Jason Brownlee, PhD. (AI) provides valuable history around the history of generating text for images and structurally explains the elements of a neural network captioning model. Brownlee also explains dominant methods prior to end-to-end neural networks and delves into the concept of feature extraction models and language models alongside their applications in the field.

Published in 2021, a blog post titled [Image Alt Text Generation Using Image Recognition](https://www.gainchanger.com/image-alt-text-generation/) is written by Dr Michaela Spiteri BEng, MSc, PhD (AI / Healthcare domain), a well-published researcher in the field of AI and machine-learning, explores the importance of alternate text, including both in the use of screen readers and SEO performance. The post also discusses varying APIs and available solutions for people who wish to generate alternate text for images on their websites.

We additionally found a useful [medium post](https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb) by Anne Bonner, which exists as the third part of a series on deep learning and importantly highlights convolutional neural networks (CNNs), which represent a breakthrough in image recognition. Bonner writes about the history and architecture of these CNNs and provides relevant examples and reader friendly walkthroughs about how CNNs work. She excellently outlines each feature of an example of convolutional neural networks including photos and code to supplement the technical writing.

The article [Image Processing using CNN: A Beginners Guide](https://www.analyticsvidhya.com/blog/2021/06/image-processing-using-cnn-a-beginners-guide/) by Mohit Tripathi provides an extended walkthrough of applications of code on the well known MNIST Dataset. Tripathi explores convolutional neural networks and goes into the specifics of how to construct, train and evaluate convolutional neural networks. Moreover, it teaches how to improve the models based on the data and how to understand and interpret the results of data training.

In terms of generating captions, we turned towards [Automatic Image Captioning Based on ResNet50 and LSTM with Soft Attention](https://www.hindawi.com/journals/wcmc/2020/8909458/), a research article which presents a joint model based on ResNet50 and LSTM with soft attention to be used for automatic image captioning. The authors utilize one encoder, adopting ResNet50, and one decoder, designed with LSTM, to create an “extensive representation of the given image” and to “selectively focus the attention over certain parts of an image to predict the next sentence”. Overall, this state-of-the-art model combines CNNs and RNNs for the most effective results in automatic image captioning.

## Some Other Implementations:
1. [PyTorch](https://github.com/ruotianluo/ImageCaptioning.pytorch) <br>
2. [TensorFlow](https://github.com/DeepRNN/image_captioning) <br>
3. [Torch](https://github.com/karpathy/neuraltalk2) <br>
4. [Torch](https://github.com/jcjohnson/densecap) <br>
5. [TensorFlow](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/image_captioning.ipynb)

## Citations
1. How to Automatically Generate Textual Descriptions for Photographs with Deep Learning
Brownlee, Jason. “How to Automatically Generate Textual Descriptions for Photographs with Deep Learning.” *Machine Learning Mastery*, 7 Aug. 2019, https://machinelearningmastery.com/how-to-caption-photos-with-deep-learning/. <br>
2.  The Complete Beginner’s Guide to Deep Learning: Convolutional Neural Networks and Image Classification
Bonner, Anne. “The Complete Beginner’s Guide to Deep Learning: Convolutional Neural Networks and Image Classification.” *Medium*, Towards Data Science, 1 June 2019, https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb. <br>
Spiteri, Michaela Tromans-Jones. “Image Alt Text Generation Using Image Recognition.” *GainChanger*, 14 Apr. 2021, https://www.gainchanger.com/image-alt-text-generation/. <br>
Yan Chu, Xiao Yue, Lei Yu, Mikhailov Sergei, Zhengkui Wang, "Automatic Image Captioning Based on ResNet50 and LSTM with Soft Attention", *Wireless Communications and Mobile Computing*, vol. 2020, Article ID 8909458, 7 pages, 2020. https://doi.org/10.1155/2020/8909458 <br>
