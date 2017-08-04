# so the basic idea of Q learning is just MDP
# given s a r s', we want to predict the Q value
# where reward = r1 + d r2 + d**2 r3 + d**3 r4 ...
# Q(s, a) = r1 + d max_a Q(s', a) (rhs: target, lhs: function q)

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

def renderImage(img):
    import matplotlib.pyplot as plt
    npimg = img.data.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()
    
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        # input is of (3, 210, 160)
        self.conv = nn.Conv2d(3, 1, kernel_size=(11, 7)) # so after is (1,200,154)
        # after max pool: (1, 100, 77)
        self.fc = nn.Linear(7700, 9) # 9 actions

    def forward(self, x):
        x = x.view(-1, 3, 210, 160)
        # check image is correct
        # renderImage(x[0])
        # exit(1)
        x = F.relu(F.max_pool2d(self.conv(x), 2))
        x = x.view(-1, 7700)
        x = self.fc(x)
        return x

def QnetLoss():
    def loss(yhat, y, actions):
        # choose yhat according ot actions yhat: n x 9 wnat n x 1
        yhat = yhat.gather(1, actions.view(-1,1))
        return nn.MSELoss()(yhat, y)
    return loss
    
class Trainer(object):
    def __init__(self, model, loss=QnetLoss(), max_epoch=30):
        self.loss = loss
        self.model = model
        self.max_epoch = max_epoch
        
    def unison_shuffled_copies(self, X, y, actions):
        assert X.shape[0] == y.shape[0]
        p = np.random.permutation(y.shape[0])
        return X[p], y[p], actions[p]

    def train(self, model, loss, actions, optimizer, x_val, y_val):
        # train one step
        x = Variable(torch.from_numpy(x_val).float(), requires_grad=False)
        y = Variable(torch.from_numpy(y_val).float(), requires_grad=False)
        actions = Variable(torch.from_numpy(actions).long(), requires_grad=False)
        # Reset gradient
        optimizer.zero_grad()
        # Forward
        yhat = model.forward(x)
        output = loss(yhat, y, actions)
        # Backward
        output.backward()
        # Update parameters
        optimizer.step()
        return output.data[0] # this really is loss

    def fit(self, X, y, actions, alpha=0.001, ):
        n_examples = X.shape[0]
        model = self.model
        optimizer = optim.SGD(model.parameters(), lr=0.001,
                              momentum=0.9, weight_decay=alpha)
        loss = self.loss
        batch_size = 500

        for i in range(self.max_epoch):
            X, y, actions = self.unison_shuffled_copies(X, y, actions)
            cost = 0.
            num_batches = math.ceil(n_examples / batch_size)
            for k in range(num_batches):
                start, end = k * batch_size, min((k + 1) * batch_size, n_examples)
                cost += self.train(model, loss, actions[start:end],
                                   optimizer, X[start:end], y[start:end])
            
            print("Epoch %d, train loss = %f" % (i + 1, cost / num_batches))

    def getModel(self):
        return self.model
