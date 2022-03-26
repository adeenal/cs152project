# We are all Monets

## Team Members
+ Adeena Liang
+ Irmak Bukey
+ Olina Wong
+ Sadie Zhao

## Introduction
Writers have their unique prose, musicians have their unique vibe, and no differently, artists have their own unique style that are recognisable in their works throughout their lives. With the development of generative adversarial networks, we can utilise neural networks to imitate any particular style of a specific artist. 

With this project, we wish to be able to replicate the style of Monet given any image. People have well-studied the typical image-to-image translation problem, where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. In this project, for example, we are given monet's paintings as our training data set, while paired training data are unavailable. Thus, we wish to achieve our goal through a recently developed architecture named Cycle-Consistent Adversarial Networks (CycleGANs), which is developed on the base of traditional GANs and aimed at solving unpaired image to image translation problem. 


## Project Goals

1. Explore methods for cleaning the training and test data sets
2. Learn and understand generative adversarial networks (GANs) and Cycle-Consistent Adversarial Networks (CycleGANs).
3. Train a neural network that is able to transfer any image to Monet style
4. Try to improve the performance of the neural network

# Literature Review (Need to be updated)
During our research process, we examined both research articles and existing implementations which relate to our project topic. Similarly, we also investigated academic literature and relevant blog postings to gain stronger understandings of how to implement our project both successfully and effectively.

In a [blog posting](https://machinelearningmastery.com/how-to-caption-photos-with-deep-learning/) focused around machine learning, Jason Brownlee, PhD. (AI) provides valuable history around the history of generating text for images and structurally explains the elements of a neural network captioning model. Brownlee also explains dominant methods prior to end-to-end neural networks and delves into the concept of feature extraction models and language models alongside their applications in the field.

Published in 2021, a blog post titled [Image Alt Text Generation Using Image Recognition](https://www.gainchanger.com/image-alt-text-generation/) is written by Dr Michaela Spiteri BEng, MSc, PhD (AI / Healthcare domain), a well-published researcher in the field of AI and machine-learning, explores the importance of alternate text, including both in the use of screen readers and SEO performance. The post also discusses varying APIs and available solutions for people who wish to generate alternate text for images on their websites.

We additionally found a useful [medium post](https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb) by Anne Bonner, which exists as the third part of a series on deep learning and importantly highlights convolutional neural networks (CNNs), which represent a breakthrough in image recognition. Bonner writes about the history and architecture of these CNNs and provides relevant examples and reader friendly walkthroughs about how CNNs work. She excellently outlines each feature of an example of convolutional neural networks including photos and code to supplement the technical writing.

The article [Image Processing using CNN: A Beginners Guide](https://www.analyticsvidhya.com/blog/2021/06/image-processing-using-cnn-a-beginners-guide/) by Mohit Tripathi provides an extended walkthrough of applications of code on the well known MNIST Dataset. Tripathi explores convolutional neural networks and goes into the specifics of how to construct, train and evaluate convolutional neural networks. Moreover, it teaches how to improve the models based on the data and how to understand and interpret the results of data training.

In terms of generating captions, we turned towards [Automatic Image Captioning Based on ResNet50 and LSTM with Soft Attention](https://www.hindawi.com/journals/wcmc/2020/8909458/), a research article which presents a joint model based on ResNet50 and LSTM with soft attention to be used for automatic image captioning. The authors utilize one encoder, adopting ResNet50, and one decoder, designed with LSTM, to create an “extensive representation of the given image” and to “selectively focus the attention over certain parts of an image to predict the next sentence”. Overall, this state-of-the-art model combines CNNs and RNNs for the most effective results in automatic image captioning.

## Methods
In this project, we plan to use Tensorflow and base our project on Kaggle notebook. Our dataset, provided by Kaggle, consists of four parts:
1. monet_jpg - 300 Monet paintings sized 256x256 in JPEG format
2. monet_tfrec - 300 Monet paintings sized 256x256 in TFRecord format
3. photo_jpg - 7028 photos sized 256x256 in JPEG format
4. photo_tfrec - 7028 photos sized 256x256 in TFRecord format

It is important to note that the input traning data are not pair images. Moreover, we would add more photos by ourselves as inputs for tests and for fun, and the outputs would be new Monet-style images based on the original input photos. If time is allowed, we would also possibly experiment with the artistic style of other artists using [the CycleGAN dataset](https://github.com/junyanz/CycleGAN).

The main architecture we plan to use in this project is CycleGAN, an approach “for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples” based on traditional Generative Adversarial Networks (GANs). Finally, we will evaluate our results based on the MiFID (Memorization-informed FID) calculated by Kaggle, which is commonly used in recent publications as the standard for evaluation methods of GANs.

## Ethics
In terms of ethics, there is the question of whether the art should be attributed to the artist of the original image or to the style of the converted art. Consequently, there is also the existing question in research of the validity of art that is created by a robot. We will seek to explore the idea of ownership in the intersection of digital property and technology. There is also the ethical topic of when using the model, the permissibility of using images that are owned by another.



## Citations
1. How to Automatically Generate Textual Descriptions for Photographs with Deep Learning
Brownlee, Jason. “How to Automatically Generate Textual Descriptions for Photographs with Deep Learning.” *Machine Learning Mastery*, 7 Aug. 2019, https://machinelearningmastery.com/how-to-caption-photos-with-deep-learning/. <br>
2.  The Complete Beginner’s Guide to Deep Learning: Convolutional Neural Networks and Image Classification
Bonner, Anne. “The Complete Beginner’s Guide to Deep Learning: Convolutional Neural Networks and Image Classification.” *Medium*, Towards Data Science, 1 June 2019, https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb. <br>
3. Spiteri, Michaela Tromans-Jones. “Image Alt Text Generation Using Image Recognition.” *GainChanger*, 14 Apr. 2021, https://www.gainchanger.com/image-alt-text-generation/. <br>
4. Yan Chu, Xiao Yue, Lei Yu, Mikhailov Sergei, Zhengkui Wang, "Automatic Image Captioning Based on ResNet50 and LSTM with Soft Attention", *Wireless Communications and Mobile Computing*, vol. 2020, Article ID 8909458, 7 pages, 2020. https://doi.org/10.1155/2020/8909458 <br>
