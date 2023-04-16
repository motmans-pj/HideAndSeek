# HideAndSeek
Group project in Reinforcement Learning Class
Authors: Paul Bédier, Lucrezia Certo, Pieter Jan Motmans
Institute: MSc in Data Sciences & Business Analytics, CentraleSupelec

/!\ PLEASE RUN NOTEBOOK USING COLAB fOR RENDER COMPATIBILITIES /!\

# About the Project
The goal of this project is two-fold. The first objective is to implement a custom environment built on the OpenAI Gym. The second one is to analyze the behavior of two agents in a multi-agent setting. The idea was taken from "Emergent Tool Use From Multi-Agent Autocurricula" by OpenAI (https://arxiv.org/abs/1909.07528). However, given the constraints in computational power, we had to simplify the game so to limit the number of states. To achieve this we decided to have the two agents operate in a limited field with a fixed number of obstacles in fixed positions.

# Files Organization
```
.
├── agents
│   *folder containing implementations of agents*
├── images
│   *folder containing images used for RGB render of the environment*   
├── utilities
│   *folder containing useful functions such as training/testing loops and plotting*
├── environment.py
│   *class implementation of the hide and seek 2D environment*
└── HideAndSeek.ipynb
    *notebook containing all our experiments /!\ PLEASE RUN USING COLAB fOR RENDER COMPATIBILITIES /!\*
```

# Requirements
gymnasium==0.28.1 <br />
matplotlib==3.6.2 <br />
numpy==1.22.4 <br />
opencv_python==4.7.0.72 <br />
PettingZoo==1.22.3 <br />
protobuf==3.20.3 <br />
pygame==2.1.2 <br />
seaborn==0.12.2 <br />
tqdm==4.64.1 <br />

# References
1. Baker, B., Kanitscheider, I., Markov, T., Wu, Y., Powell, G., McGrew, B., Mordatch, I.: Emergent tool
use from multi-agent autocurricula. arXiv preprint arXiv:1909.07528 (2019)
2. Christopher J.C.H. Watkins, P.D.: Technical note: Q-learning. Machine Learning, 8, 279-292 (1992)
3. G. A. Rummery, M.N.: Modified connectionist q-learning. Online Q-Learning using connectionist sys-
tems, 6 (1994)
4. Richard S. Sutton, A.G.B.: Episodic semi-gradient sarsa. Reinforcement Learning: An Introduction,
chapter 10.1 (2018)

