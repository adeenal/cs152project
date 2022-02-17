# Image Captioning

## Team Members
+ Adeena Liang
+ Irmak Bukey
+ Olina Wong
+ Sadie Zhao

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

## Related Works Search
1. How to Automatically Generate Textual Descriptions for Photographs with Deep Learning
2. The Complete Beginner’s Guide to Deep Learning: Convolutional Neural Networks and Image Classification
3. Deep convolution neural network for image recognition
4. Image Processing using CNN: A beginners guide
5. Artificial Neural Networks in Image Processing for Early Detection of Breast Cancer
6. Image Alt Text Generation Using Image Recognition
7. Chart-Text: A Fully Automated Chart Image Descriptor
8. Chinese alt text writing based on deep learning
9. Text Extraction in Python with Neural Networks: Deep Learning for Image Processing
10. Facial Recognition Neural Networks Confirm Success of Facial Feminization Surgery
11. Automatic Image Captioning Based on ResNet50 and LSTM with Soft Attention
12. Image Captioning with Compositional Neural Module Networks
13. Diverse Image Captioning with Context-Object Split Latent Spaces


1. How to Automatically Generate Textual Descriptions for Photographs with Deep Learning
Brownlee, Jason. “How to Automatically Generate Textual Descriptions for Photographs with Deep Learning.” Machine Learning Mastery, 7 Aug. 2019, https://machinelearningmastery.com/how-to-caption-photos-with-deep-learning/. 
In this post from his blog focused around machine learning, Jason Brownlee, PhD. (AI) provides valuable history around the history of generating text for images and structurally explains the elements of a neural network captioning model. He additionally helpfully cites and links other articles and papers published in the last five years that relate to image caption generators. Brownlee also explains dominant methods prior to end-to-end neural networks and delves into the concept of feature extraction models and language models alongside their applications in the field.

2.  The Complete Beginner’s Guide to Deep Learning: Convolutional Neural Networks and Image Classification
Bonner, Anne. “The Complete Beginner’s Guide to Deep Learning: Convolutional Neural Networks and Image Classification.” Medium, Towards Data Science, 1 June 2019, https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb. 
This medium post is the third part of a series on deep learning which highlights convolutional neural networks (CNNs) which represent a breakthrough in image recognition. Bonner writes about the history and architecture of these CNNs and provides relevant examples and reader friendly walkthroughs about how CNNs work. She excellently outlines each feature of an example of convolutional neural networks including photos and code to supplement the technical writing.
6. Image Alt Text Generation Using Image Recognition
Spiteri, Michaela Tromans-Jones. “Image Alt Text Generation Using Image Recognition.” GainChanger, 14 Apr. 2021, https://www.gainchanger.com/image-alt-text-generation/. 
Published in 2021, this blog post is written by Dr Michaela Spiteri BEng, MSc, PhD (AI / Healthcare domain), who is a well-published researcher in the field of AI and machine-learning. She explores the importance of alternate text, including both in the use of screen readers and SEO performance. The post also discusses varying APIs and available solutions for people who wish to generate alternate text for images on their websites.
11. Automatic Image Captioning Based on ResNet50 and LSTM with Soft Attention
Yan Chu, Xiao Yue, Lei Yu, Mikhailov Sergei, Zhengkui Wang, "Automatic Image Captioning Based on ResNet50 and LSTM with Soft Attention", Wireless Communications and Mobile Computing, vol. 2020, Article ID 8909458, 7 pages, 2020. https://doi.org/10.1155/2020/8909458

This research article published in 2020 presents a joint model based on ResNet50 and LSTM with soft attention to be used for automatic image captioning. The authors utilize one encoder, adopting ResNet50, and one decoder, designed with LSTM, to create an “extensive representation of the given image” and to “selectively focus the attention over certain parts of an image to predict the next sentence”. Overall, this state-of-the-art model combines CNNs and RNNs for the most effective results in automatic image captioning.


Some other Implementations:
1) PyTorch: https://github.com/ruotianluo/ImageCaptioning.pytorch
2) TensorFlow: https://github.com/DeepRNN/image_captioning
3) Torch: https://github.com/karpathy/neuraltalk2
4) Torch: https://github.com/jcjohnson/densecap
5)TensorFlow: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/image_captioning.ipynb
