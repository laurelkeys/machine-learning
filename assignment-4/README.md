# Final Project
Following one of the (⭐⭐⭐) problems presented by [OpenAI](https://openai.com/) in [**Requests for Research 2.0**](https://openai.com/blog/requests-for-research-2/), we aim to explore the task of **Transfer Learning Between Different Games via Generative Models**.

## Transfer Learning Between Different Games via Generative Models
> Proceed as follows:
> - Train 11 good policies for 11 [Atari](https://github.com/openai/gym#atari) games.  
    Generate 10,000 trajectories of 1,000 steps each from the policy for each game.
> - Fit a generative model (such as the [Transformer](https://arxiv.org/abs/1706.03762)) to the trajectories produced by 10 of the games.
> - Then fine-tune that model on the 11th game.
> - Your goal is to quantify the benefit from pre-training on the 10 games.  
    How large does the model need to be for the pre-training to be useful?  
    How does the size of the effect change when the amount of data from the 11th game is reduced by 10x? By 100x?

## Resources
- Gym
  - [Getting Started with Gym](https://gym.openai.com/docs/)
- Stable baselines
  - [[GitHub](https://github.com/hill-a/stable-baselines/tree/master)] [[Docs](https://stable-baselines.readthedocs.io/en/master/)] [[Examples](https://github.com/hill-a/stable-baselines/blob/master/docs/guide/examples.rst)]
  - [Stable Baselines: a Fork of OpenAI Baselines — Reinforcement Learning Made Easy](https://towardsdatascience.com/stable-baselines-a-fork-of-openai-baselines-reinforcement-learning-made-easy-df87c4b2fc82)
- Attention and the Transformer model
  - TensorFlow's tutorial: [Transformer model for language understanding](https://www.tensorflow.org/tutorials/text/transformer)
  - Suggested (implemented) models for image generation on [tensor2tensor](https://github.com/tensorflow/tensor2tensor#image-generation)
  - [Image Transformer](https://ai.google/research/pubs/pub46840/) and [Sparse Transformer](https://openai.com/blog/sparse-transformer/)
  - Lilian Weng's blog post: [Attention? Attention!](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)
- Reinforcement learning
  - Lilian Weng's blog post: [Policy Gradient Algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html)
  - Andrej Karpathy's blog post: [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)
  - [David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)'s [course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ)
  - [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
