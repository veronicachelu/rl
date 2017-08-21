# Categorical DQN.

Implementation of the Categorical DQN introduced in A distributional Perspective on Reinforcement Learning.

Alongside is an implementation of the standard DQN.

Code is not optimized...yet.

### System requirements

* Python3.5

### Python requirements

Install the game of Catcher:

    git clone https://github.com/ioanachelu/gym_fast_envs
    cd gym_fast_envs
    pip install -r requirements.txt
    pip install -e .
    
### Options

* You can run basic DQN or CategoricalDQN. General flags can be found in ```configs/base_flags.py```. 
You can edit this file or use  ```-algorithm="DQN"``` or ```-algorithm="CategoricalDQN"```
* For specific algorithm parameter consult ```configs/categorical_dqn_flags.py``` and ```configs/dqn_flags.py```

### Training and resuming

* To train use:

        python run.py
        
        python run.py --resume=True
        
* To see training progress run tensorboard from the ```summaries/CategoricalDQN``` or ```summaries/DQN``` directory:
       
       tenorboard --logdir=.

### TODO:
- [ ] Add learning rate schedule
- [ ] Add evaluation procedure
- [ ] Better switch between agents
- [ ] Add result plots
- [ ]Atari

### Training Results


