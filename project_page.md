# We are all Monets

## Team Members
+ Sadie Zhao
+ Adeena Liang
+ Irmak Bukey
+ Olina Wong

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
In this project, we plan to use Tensorflow and base our experiment on both Kaggle notebook (with TPU V3-8 as an accelerator) and Google Colab (with GPU as an accelerator), which allow us to compare and evaluate the runtime performance of our project. 

Our dataset, provided by Kaggle, consists of four parts:
1. _monet_jpg_ - 300 Monet paintings sized 256x256 in JPEG format
2. _monet_tfrec_ - 300 Monet paintings sized 256x256 in TFRecord format
3. _photo_jpg_ - 7028 photos sized 256x256 in JPEG format
4. _photo_tfrec_ - 7028 photos sized 256x256 in TFRecord format

The _monet_tfrec_ and _monet_jpg_ directories contain the same painting images, and the _photo_tfrec_ and _photo_jpg_ directories contain the same photos. In our experiment, we load and process the data from _monet_tfrec_ and _photo_tfrec_ to create our dataset: we decode the tfrec file to jpeg file, resize all the files to three channel image of size [256, 256, 3], and zip all of them to a dataset. 

It is important to note that the input training data are not pair images. That is, for each monet image provided, there is no corresponding photo image; and for each real photo image, there is no corresponding monet image. The unpaired nature of our  training data promotes us to employ the Cycle-Consistent Adversarial Networks (CycleGAN). CycleGAN is a recently-developed approach “for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples”. Traditional image-to-image translation solution based on Conditional Adversarial Networks depends on the availability of training examples where the same data is available in both domains. That is, say f: X->Y is a map from a source domain X to a target domain Y. Then, the training sample for Conditional Adversarial Networks based image-to-image translation model should looks like (x, f(x)) pair. However, CycleGAN eliminated the need for a paired image by making a two-step transformation of the source domain image - first by trying to map it to the target domain and then back to the original image. (More details of CycleGAN will be discussed in discussion section)
![paired_image_input](/cs152project/img/unpaired_img_translation.jfif)


## Discussion Outline
The data we will present will be a series of images in various file sizes that have been converted to “Monet” style images. The model will have been trained on Monet style images, and we will interpret our output prediction images using a specific set of parameters known as MiFID (Memorization-informed Fréchet Inception Distance), which is a modification from Fréchet Inception Distance (FID). We will specifically derive our results by submitting our output images to a evaluation model of MiFID found on Kaggle and analyse the output score to determine accuracy of our model. Our work is similar to other models in that various types of image transformation models exist, such as image filters and image classification. One such example is a project from two semesters ago done by students who attempted to create a General Adversarial Network (GAN) to output a Picasso style artwork. We will also examine how different variants of GANs (like traditional GANs, DCGANs, and CycleGANs) perform and which one would best suit our desired goals. We will prove our claims by showing the accuracy of our models and its visualisations to audiences. On the other hand, we will provide evidence supporting our points regarding the ethical concerns of such models and how they can be combated. 

Important concept: 
The GAN model architecture involves two sub-models: a generator model for generating new examples and a discriminator model for classifying whether generated examples are real (from the domain) or fake(generated by the generator model). The generator model g:R^n -> X takes a fixed-length random vector as input and generates a sample in the domain. The vector is drawn randomly according to a Gaussian distribution, and we use it as a seed for our generative process. The discriminator model d: X -. {0,1} takes an example from the domain as input (real or generated) and predicts alabel of real or fake (generated). The generator and discriminator will be trained together. Generator will generate a batch of samples, feeding into discriminator along with real samples. The discriminator will be updated to better discriminate fake and real sample, while the generator will then be updated based on how well it fools the discriminator. 

The CycleGAN is an extension of the GAN architecture that involves the simultaneous training of two generator models and two discriminator models. We can summarize it as:
1. Generator Model 1:
  * Input: Takes input form collection 1.
  * Output: Generates a sample in collection 2.
2. Discriminator Model 1:
  * Input: Takes real samples from collection 2 and output from Generator Model 1.
  * Output: Likelihood of a sample is from collection 2.
  
3. Generator Model 2:
  * Input: Takes input form collection 2.
  * Output: Generates a sample in collection 1.
4. Discriminator Model 2:
  * Input: Takes real samples from collection 1 and output from Generator Model 2.
  * Output: Likelihood of a sample is from collection 1.


## Ethics
**Is this project ethical?**

We see this project as generally ethical as long as the model is used with good and honest intentions. While the model could be used to generate counterfeit art, the model also offers positive opportunities for learning about art styles and different kinds of neural networks and how AI and art can be interconnected.
We are aware of the risks of a model that could be used nefariously but believe that the level at which we seek to apply it has more value for educational purposes and will not reach an audience with malicious intent.

**What parts of the project have historically faced ethical concerns?**

When art is generated by or through technology, there tends to be the question of did the technology itself create the art? While it could be attributed to the person who built the technology, more often than not is the technology based on another person’s art style. In fact, technology allows artist to create art in new ways, as stated in the article Computers Do Not Make Art, People Do where the author mentions how in their research that, “seeing an artist create something wonderful with new technology is thrilling, because each piece is a step the evolution of new forms of art” (Hertzmann). On the other hand, there exists ethical concerns over the ownership of the art and whether one’s images can be used in the model (either as an input source or a style basis), but this can be best resolved by only using images where permission has been granted.

**How diverse is our team?**

Our project team is diverse and comes from different backgrounds and identities. We all have varying levels of experience and background with numerous forms of art and modelling. We hope to apply our knowledge to the best of our abilities in this project and will use our different experiences to make a well rounded model with the least amount of bias possible.

**Is our data valid for its intended use? Could there be bias in our dataset?**

Our data is valid and would not contain bias as we are using a collection of many different images. Although some of the images could be more easily turned into art than others, our model does not predict things such as statistics or outcomes but rather transforms images. Consequently, we see less of an ethical concern with bias in the data and more on the side of the outcome of the model.

**How might we negatively affect individuals’ with the model?**

While we are using the image style of a certain artist, we are not attempting to take credit for or plagiarise their work. Moreover, we recognize how the original art should be attributed to the artist of the original image and how the original images we input into the model to be converted still belong to the original owner. There exists the question of who the converted art belongs to as it exists in a grey area between the artist style, the original image owner, and the user of the model.

## Reflection
In this project, we've studied how to accomplish complicated image-to-image translation jobs through the newly-developed CycleGAN architecture. However, our way of implementing the CycleGAN model is not polished, and there are a lot of approaches to improve the performance of our model. These could be interesting topics for continuing works. 

1. In our experiment, we only built a basic CycleGAN model with rudimentary generators and discriminators. They did work, but we didn’t spend enough time on improving their performance. In fact, if we have more time, we can definitely test more on the parameters and neural network layers. Moreover, in real-life projects and scientific research, there are a huge amount of known and emerging approaches to improve the performance of GANs. For example, one can do feature matching, minibatch discrimination, one-sided label smoothing, or historical averaging. Some of these techniques may also be applied to improving the behavior of our CycleGAN model, and investigating which of them work well can be an interesting research topic. 

2. Besides the CycleGAN model itself, we can also improve our results by looking into this specific question. For example, we mentioned in our discussion section that the effectiveness of the model is largely affected by the traits of the original input photo, including image brightness, saturation, contrast. Thus, if we could study more on computer vision and learn more about how to deal with these image arguments, we would be able to better conquer the problem caused by extreme image augments. 

Next time, we would start coding earlier, so that I could have more time exploring and improving the final results. 

## Citations
1. How to Automatically Generate Textual Descriptions for Photographs with Deep Learning
Brownlee, Jason. “How to Automatically Generate Textual Descriptions for Photographs with Deep Learning.” *Machine Learning Mastery*, 7 Aug. 2019, https://machinelearningmastery.com/how-to-caption-photos-with-deep-learning/. <br>
2.  The Complete Beginner’s Guide to Deep Learning: Convolutional Neural Networks and Image Classification
Bonner, Anne. “The Complete Beginner’s Guide to Deep Learning: Convolutional Neural Networks and Image Classification.” *Medium*, Towards Data Science, 1 June 2019, https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb. <br>
3. Spiteri, Michaela Tromans-Jones. “Image Alt Text Generation Using Image Recognition.” *GainChanger*, 14 Apr. 2021, https://www.gainchanger.com/image-alt-text-generation/. <br>
4. Yan Chu, Xiao Yue, Lei Yu, Mikhailov Sergei, Zhengkui Wang, "Automatic Image Captioning Based on ResNet50 and LSTM with Soft Attention", *Wireless Communications and Mobile Computing*, vol. 2020, Article ID 8909458, 7 pages, 2020. https://doi.org/10.1155/2020/8909458 <br>
