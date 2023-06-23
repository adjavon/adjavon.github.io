---
published: false
layout: post
title: Uncertain spice
tags: deep-learning uncertainty classifiers benchmark
---
# And if you <span style="color:#73BFF9">don't know</span>, now you <span style="color:#EF9A90">know</span>

TL;DR: My notes on: A survey of Uncertainty in Deep Neural Networks, with a spicy little benchmark.

## Skippable background
Recently, I was asked to train a classifier on a problem that I fully expected would be an easy problem.
It was a large dataset, a two class problem, and there was a known signal for the positive class.
To my surprise, however, the first few tests I ran ended up performing almost exactly randomly.
After digging a bit into the data, I realized that although there was a known signal for the positive class, that known signal didn't appear in all of the positively labelled sampled.
In other words, many of the images that I was getting were uninformative.

What was annoying about the network[^1] wasn't that it was wrong, it was that it was confidently wrong.
This struggle got me thinking about ways that I could get my network to just admit when it didn't know.
This prompted a deep[^2] dive into uncertainty estimation.
This post describes what I did to think about it.

Code is [here](https://github.com/adjavon/uncertain-spice/tree/main), I welcome comments, thoughts, ideas, etc.

Credit assignment: I did my best! Figures that are not mine are taken from the paper, and I've marked them as such in their captions.
Code that was taken from elsewhere is marked directly in the file, in a comment, but I'm mostly grateful to [this repo](https://github.com/dougbrion/pytorch-classification-uncertainty/) for the EDL implementation.


## Types of uncertainty
The uncertainty literature seems to distinguish between two types of uncertainties: data (or aleatoric) uncertainty and model (or epistemic) uncertainty.
Model uncertainty can usually be fixed with more data.

Out-of-distribution detection is a sub-task related to model uncertainty.
When a neural network sees something that it has never seen before, we should not expect it to give a (reasonable) output.
This can theoretically be solved with more data, but in practice that would require training with all possible out-of-distribution samples[^3].


Data uncertainty is a harder nut to crack, and was what I thought I was dealing with.
It is inherent in the data, so it cannot be fixed with more data.
Even though it can't be fixed, learning to quantify it is immensely eye-opening already.
It helps trouble-shoot (is my model not learning at all, or is it learning that there isn't anything to learn?), filter out points (how confident is this prediction?), and even potentially learn something about the data (namely, its inherent stochasticity).

<figure>
    <img src='/assets/images/uncertain_spice/types_of_uncertainty.png'>
    <figcaption> Paper figure: Types of uncertainty according to our main reference. My ideal uncertainty estimation method, in this exploration, would be able to separate between the three. Most importantly, however, I wanted to be able to recognize data/aleatoric uncertainty and differentiate it from out-of-distribution samples.</figcaption>
</figure>

## The data
As anybody who has worked with me long enough will tell you, I usually learn these kinds of concepts by generating a silly synthetic dataset, trying a bunch of techniques as quickly as possible, and accumulating anecdotal evidence and gut feelings. For this project, the task was determining how spicy an image was, based on the amount of pepper in it.

<table>
    <tr>
        <td>Juicy</td>
        <td>Mild</td>
        <td>Spicy</td>
        <td>OOD</td>
    </tr>
    <tr>
        <td><img src='/assets/images/uncertain_spice/juicy.png' width='100'></td>
        <td><img src='/assets/images/uncertain_spice/mild.png' width='100'></td>
        <td><img src='/assets/images/uncertain_spice/spicy.png' width='100'></td>
        <td><img src='/assets/images/uncertain_spice/hot.png' width='100'></td>
    </tr>
</table>

The data was synthetically generated from a set of vegetable templates.
I randomly picked a number of each template, scale and rotate them, and put them together onto a plain background.
Whichever vegetable is most represented defines the class of the image as a whole.

I also created out-of-distribution images by including an emoji in the testing set that was not see in the training set.
I chose the sun because the sun is hot, but not necessarily spicy[^4].

Finally, I created uninformative[^5] samples but choosing two of the vegetables and putting the same number of each in the image.
Ideally, this should return a 50/50 split between the two options.
The hope, here was that if the model had understood the concept (that the number of items of each type defined the class), then it would output a very low uncertainty for this 50/50 split.


<table>
    <tr>
        <td>Informative</td>
        <td>Juicy<br><img src='/assets/images/uncertain_spice/informative_juicy.png' width='100'></td>
        <td>Mild<br><img src='/assets/images/uncertain_spice/informative_mild.png' width='100'></td>
        <td>Spicy<br><img src='/assets/images/uncertain_spice/informative_spicy.png' width='100'></td>
    </tr>
    <tr>
        <td>OOD(but informative)</td>
        <td>Juicy<br><img src='/assets/images/uncertain_spice/ood_juicy.png' width='100'></td>
        <td>Mild<br><img src='/assets/images/uncertain_spice/ood_mild.png' width='100'></td>
        <td>Spicy<br><img src='/assets/images/uncertain_spice/ood_spicy.png' width='100'></td>
    </tr>
    <tr>
        <td>Uninformative</td>
        <td><img src='/assets/images/uncertain_spice/uninformative_1.png' width='100'></td>
        <td><img src='/assets/images/uncertain_spice/uninformative_2.png' width='100'></td>
        <td><img src='/assets/images/uncertain_spice/uninformative_3.png' width='100'></td>
    </tr>
</table>

## The benchmark
The baseline model that I used was a 3-class VGG, as implemented [here](https://github.com/funkelab/funlib.learn.torch/blob/master/funlib/learn/torch/models/vgg2d.py)
- `fmaps=32`
- `output_classes=3`
- `input_fmaps=3`

I trained for 1000 iterations, with a batch size of 16, with a Cross Entropy loss and an Adam optimizer.

The questions that I wanted to answer each time were:
- How are the templates classified?
- What is the in-distribution vs out-of-distribution accuracy?
- How can I estimate the uncertainties for:
    - informative samples
    - uninformative samples
    - out-of-distribution samples

The hope was to get a high accuracy for both in-distribution and out-of-distribution samples, but with a much higher uncertainty for out-of-distribution samples.

## The methods
<figure>
    <img src='/assets/images/uncertain_spice/model_types.png'>
    <figcaption>Paper figure: The different families of models. I tried an example of each, picking the easiest/quickest to implement for each. The order of operations in the text is D, B, C, A. My favorite was A.</figcaption>
</figure>

## Test-time augmentation
What it does
Addresses epistemic uncertainty using data augmentation
No need to re-train
Augmentations can be user-defined or learned [1]

What I did
Right-angle rotations and flips
Mean of softmax(logits)
Std of softmax(logits)

## Bayesian method - Dropout
What it does
Addresses epistemic uncertainty using probabilistic model weights
Special training necessary

What I did
Dropout in the dense layers of VGG (test and train)
10 forward passes for each sample
Mean and Std of softmax(logits)

## Ensemble method - Random initialization
What it does
Addresses epistemic uncertainty using a variety model weights
Special training necessary

What I did
Dropout in the dense layers of VGG (test and train)
10 forward passes for each sample
Mean and Std of softmax(logits)

## Interlude - The Dirichlet distribution

This is the figure that convinced me to fully read the paper.
<figure>
    <img src='/assets/images/uncertain_spice/dirichlet.png'>
    <figcaption>Paper figure: The Dirichlet distribution visualized for the three types of data that I created earlier: informative, uninformative, and out-of-distribution. </figcaption>
</figure>

## Evidential neural networks

## Footnotes
[^1]: This is also true of many other networks that I've trained, to be honest...
[^2]: But not that deep
[^3]: Think, adding an 'none' class to a classifier output and training with examples of out-of-distribution or un-annotated samples. Also, in this case I would be nit-picky and argue that these samples are now in-distribution, since they are in the training set.
[^4]: I allegedly said this when I presented this in a journal club.
[^5]: But technically in-distribution?
