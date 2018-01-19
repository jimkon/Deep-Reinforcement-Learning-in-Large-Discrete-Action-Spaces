# Deep-Reinforcement-Learning-in-Large-Discrete-Action-Spaces
Link to [paper](https://arxiv.org/abs/1512.07679)

Implementation of the algorithm in Python 3, TensorFlow and OpenAI Gym.



This paper introduces Wolpertinger training algorithm that extends the Deep Deterministic Policy Gradient training algorithm introduced in [this](https://arxiv.org/abs/1509.02971) paper.

I used and extended  **stevenpjg**'s implementation of **DDPG** algorithm found [here](https://github.com/stevenpjg/ddpg-aigym) licensed under the MIT license.

The agent that is presented on the paper needs discrete action space to work. My implementation is only for continuous action spaces that he discretize, but is easily modifiable through the constructor function of the agent where the _self.action_space_ is set.
