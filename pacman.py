import gym, os
from Q_learning import Qnet, Trainer
import torch
from torch.autograd import Variable
import numpy as np
import copy
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
# load previous model
if os.path.exists('qnet.pt'):
    Q = torch.load('qnet.pt')
else:
    Q = Qnet()

# build a dataset
def gen_random_sars(n_episode=1, n_frame=1000, close=False):
    sars = []
    burn_in = 50
    for i_episode in range(n_episode):
        s = env.reset()
        s = np.swapaxes(np.swapaxes(s,0,2), 1, 2)
        for t in range(burn_in + n_frame):
            env.render(close=close) # render the play
            a = env.action_space.sample() # 9 actions
            s_prime, r, done, info = env.step(a)
            s_prime = np.swapaxes(np.swapaxes(s_prime,0,2), 1, 2)
            if t > burn_in:
                sars.append((s.flatten(), a, r, s_prime.flatten()))
            s = s_prime
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    return sars

def sars2xya(sars, Qtarget):
    actions = np.array(list(map(lambda sars: sars[1], sars)))
    X = np.array(list(map(lambda sars: sars[0], sars)))
    s_primes = Variable(torch.from_numpy(np.array(list(map(lambda sars: sars[3],
                                                           sars)))).float())
    rewards = Variable(torch.Tensor(list(map(lambda sars: sars[2], sars))))
    target = rewards + torch.max(Qtarget(s_primes), 1)[0]

    # X, target, actions are numpy arrays
    return X, target.data.numpy(), actions

# run the model
def run(Qnet, n_episode=1, n_frame=1000, close=False):
    sars = []
    burn_in = 0
    for i_episode in range(n_episode):
        s = env.reset()
        s = np.swapaxes(np.swapaxes(s,0,2), 1, 2) # to get channels right
        for t in range(burn_in + n_frame):
            env.render(close=close) # render the play
            s_feed = Variable(torch.from_numpy(s.flatten()).float().unsqueeze(0))
            a = torch.max(Qnet(s_feed), 1)[1].data[0][0]
            s_prime, r, done, info = env.step(a)
            s_prime = np.swapaxes(np.swapaxes(s_prime,0,2), 1, 2)
            if t > burn_in:
                sars.append((s.flatten(), a, r, s_prime.flatten()))
            s = s_prime
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    return sars

def main():
    # get dataset
    sars = gen_random_sars(close=True)
    X, y, actions = sars2xya(sars, Q)

    # learn a model
    Qnew = copy.deepcopy(Q)
    t = Trainer(Qnew, max_epoch=30)
    t.fit(X, y, actions)

    # save new model
    torch.save(t.getModel(), 'qnet.pt')

if __name__ == '__main__':
    main()
    run(Q)
    
# debug
# from Q_learning import renderImage
# def renderNPImg(img):
#     img = Variable(torch.from_numpy(img))
#     img = img.view(3,210,160)
#     renderImage(img)

# for i in range(X.shape[0]):
#     renderNPImg(X[i])
#     print(y[i])
