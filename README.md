# HideAndSeek

## Current stage

We currently have a Q-agent and an expected sarsa agent that are learning, there is a function value approximation agent for which the training function is not yet implemented (this should not take very long though). 
The environment logic seems to be working as well. 

As observations, both agents get each others' location, the obstacles are fixed so as to limit the number of total possible states. 
By giving the agents each others' location, we change the game to more of a game of 'tag', this is done to observe more interesting behavior. 

Problem in this setting is that it is difficult to show state value plots, and to have an objective metric for the learning. 
To solve that, we will start a stage 1 in which the hider is static and always in the same position. There is then a limited number of states, and we should be able to observe the learning quite quickly. This will allow us to make some plots and show a proof of concept. 

## Left to do

* First, we have to perform additional experiments with the agents we have implemented, Lucrezia has done Expected Sarsa and QLearning, ideally we also compare them against random seeker vs. agent hider,... and the function value approximation agent as well. What can we get from these experiments? We can see whether they learned any interesting behavior by rendering some full games. We can also compare the proportion of games won by each different agent as hider or seeker. 

* Second, I am fine with the rendering as is, it is definitely only a proof of concept but it already shows something. The orientation of the agents is for example not yet really consistent and the 'fire' coming out of the dragons mouth doesn't change orientation yet, should be quick fixes. Flash warning though, the screen when run locally flickers from frame to frame, observing the behavior in colab would be easier. 

* Third, we can either perform some sensitivity analysis on the parameters of current agents (expected sarsa, qlearning and function value approximation) or we can implement a fourth agent (for example Reinforce from lab 6). I am fine with both!

* Fourth, I will start doing the seeker going to a static, fixed location and retrieving some insights from that. 
