---
published: true
layout: post
title: Velocity Learning
---

Recently while reading [this book](https://arxiv.org/abs/2104.13478) I stumbled upon the phrase:

> Crucially, learning a dynamical system by modeling its velocity turns out to be much easier than learning its position directly. In our learning setup, this translates into an optimisation landscape with more favorable geometry, leading to the ability to train much deeper architectures than was possible before.
>
> - *Geometric Deep Learning Grids, Groups, Graphs, Geodesics, and Gauges*, p.73

It was in a short discussion about why ResNets work well and it struck me as an interesting point, although I have little intuition for why it is true.
Since reading that, I have been thinking about how this might apply in other architectures and training setups.
Why do diffusion models work so well? In the Ho et al. paper [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), they reparametrize the reverse process of the diffusion model[^1] to train the network to learn the perturbation[^2] rather than the signal. In some sense, this is modelling the velocity[^3]. They find that it performs at least as well, and in some cases better, than the more intuitive formulation before reparametrizing.

I get a similar feeling from transformers, although not as clear. The query, key, and value in an attention layer are all derived from the input data. The dot product between them means that at the output of an attention layer we have an output that is a higher-order representation of the input[^4]. This is something that common operations such as convolutions of fully connected layers don't allow, and none of the popular non-linearities used as activations provide this either. Granted, higher order description is not the same as learning velocity, but I feel like it is worth exploring[^5].

[^1]: Which is the part where they take a noisy image and turn it into a slightly less noisy image
[^2]: Here, the noise added at a given step in the forward process
[^3]: Or at least a difference, let's say those are the same in this case.
[^4]: Is it third order? I still struggle to understand the respective roles of query, key and value. They all seem like embeddings to me.
[^4]: Especially when transformers might be doing gradient descent on the context? See [Transformers learn in-context by gradient descent](https://arxiv.org/abs/2212.07677) by von Oswald et al.
