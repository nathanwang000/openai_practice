import gym
from Q_learning import Qnet, Trainer
'''
step returns 4 values

1) observation (eg. camera pixel for pac man (210,160,3))
2) reward (real value reward for previous action)
3) done (boolean: whether shold reset)
4) info (eg. number of lives left)

for actions:
0: forward
1: right
2: back ?
3:
'''

env = gym.make('MsPacman-v0')

# build a dataset
for i_episode in range(2):
    observation = env.reset()
    print(observation)
    break
    for t in range(1000):
        env.render() # render the play
        action = env.action_space.sample() # 9 actions
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


