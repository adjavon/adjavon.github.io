---
published: true
layout: post
tags: unfiltered
last_updated: 2025-03-13
description: A question I get asked (yearly, mostly by family) is ''what even does a machine learning researcher do on a daily basis?'' Here's a sneak peek. 

---
# A day in the life

A question I get asked (yearly, mostly by family) is "what even does a machine learning researcher do on a daily basis?"
In the spirit of [a lovely newsletter by Robin Sloan](https://www.robinsloan.com/newsletters/golden-door/) suggesting that we all lower our activation energy for sharing what we know, here's a sneak peek. 

Let's assume that I'm starting a new project, or a new step in a project, on this average day.
Before anything else, I need to determine what my task is, so that I can figure out what parts I need to put together to make it happen. 
Today, it's an image translation task: I want to train a neural network that takes in images of type A (bright field images of yeasts in a trap, let's say) and turns them into images of type B (the same yeasts in fluorescence). 

Obviously, I'll need a way to load the images of type A and B togethe -- this is my own data, so I'll have to dig into code written by myself and my colleagues to find the best way to do this. Once I've loaded the images, I'm looking at them and asking a few questions about them: How big are the images? How many pairs do I have? What is the range of pixel values in a given image, or across the whole set of images? 

By default (or is it prior knowledge? experience?) when doing an image-to-image task I want to use a [UNet](). 
The UNet is a popular enough network that I know I can find a good implementation of it without having to write one myself. 
For example, I can use [the delightful UNet package](https://github.com/dlmbl/dlmbl-unet) written for [the Deep Learning at MBL course](https://github.com/dlmbl/DL-MBL-2024) that I've TA'd at. 
I know who wrote it, so I'm confident it's good! 

Next I'll usually write down some expectations, and design decisions for the experiment. My loss function (MSE), how I plan to normalize my images (very carefully, in a data-dependent way because these `uint16` images have a `uint8` range at best, but the values are not consistently where you think they should be), what metrics am I going to use to determine whether I'm doing a good job (MSE, correlation, but on validation data).

This thought process takes longer when I'm doing a task I'm not used to, or worse a task I'm not sure is even possible. 
For my usual classification task, the above is 5 minutes top. 
For something new and unusual, I might have to spend some time looking up the relevant litterature, reading through some other people's code, or phoning a friend for advice. 

Once I have a rough sketch of the different pieces that need to come together, it's time to setup my coding environment. 
I usually will start a fresh `mamba` environment, to keep my projects separate. 
I'll install what dependencies I need -- this usually means running into dependency issues and debugging for a little while. 
Sometimes there are dependencies I can just cut if they don't work, sometimes I don't have a choice and I have to make them play nicely. 
Either way, eventually (minutes to days of work!) I'll have a working environment that I can use.

Then it's time put together the puzzle pieces.
There's some code to write, but usually not that much here when I'm just starting off. 
My standard order of operations is: 
1. Loading the data into the format my model will need (including augmentations!)
2. Creating the model
3. Writing a training loop, with checkpointing etc.
4. Logging things (either to WandB, or to Tensorboard, in my case)
5. Validation! I like to do validation synchronously with training because I am impatient, and because I work on a lot of "it might just not work"-type tasks

The goal, here, is to be able to get some prototype code up and running pretty quickly so that I can catch any glaring mistakes. 
Mistakes that I catch once I have something runnning are: normalization errors, image and model outputs in differentr anges, augmentations too crazy, learning rate too crazy, ... etc.

Then, once I'm convinced via prototyping that the task *is* possible with my current setup, I actually (fully!) train the model!
This is one of the more difficult parts, to me, because I have to physically detach myself from the screen to avoid watching the losses as it trains and jumping to conclusions. 
Generally, I try to do my big runs overnight, or over a meal, so that I'm away from my computer and distracted.

Once training is done I'll examine the validation results, diagnose issues, and repeat. 