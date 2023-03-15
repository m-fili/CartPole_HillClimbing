# Solving CartPole Problem using Policy-based Methods

<img src="Images/CreatedGif.gif" height="450">

### 1. Introduction
In this project we use policy-based reinforcement learning methods to
solve the cartpole problem. The goal is to keep the cartpole balanced. At
any time, we can move the cartpole to the left or right. The state space
has 4 dimensions: 
* position of cart
* velocity of cart
* angle of pole 
* rotation rate of pole.



### 2. Methods
To solve this problem, a policy-based approach has been used. Four gradient-free
optimization methods are applied to estimate the optimal policy directly. These
models are:
* Hill Climbing
* Steepest Ascent Hill Climbing
* Simulated Annealing
* Adaptive Noise Scaling

### 3. Implementation
To train an agent to keep the cartpole in balance, please follow the instructions
in `Navigation.ipynb`.


### 4. Results
The results for all four methods are shown below:


<img src="Images/vanilla_hill_climbing.png" height="250" title="Vanilla Hill Climbing">

<img src="Images/steepest_ascent.png" height="250" title="Steepest Ascent">

<img src="Images/simulated_annealing.png" height="250" title="Simulated Annealing">

<img src="Images/adaptive_noise_scaling.png" height="250" title="Adaptive Noise Scaling">

