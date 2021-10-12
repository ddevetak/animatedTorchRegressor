import torch
from torch.autograd import Variable  # Variable is now deprecated and tensors could have gradients too…!!!!!!!!!!!
import torch.nn.functional as F
import torch.utils.data as Data
from matplotlib.animation import FuncAnimation

import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np

#########################################
# global 

torch.manual_seed(1)    # reproducible
x = torch.unsqueeze(torch.linspace(-math.pi, math.pi, 1000), dim=1)
y = torch.sin(x**2) + 0.3*torch.rand(x.size())   
#y = torch.cos(x**3) + 0.3*torch.rand(x.size())   

predictionFull = []
lossFull = []

fig, ax = plt.subplots(figsize=(12,7))
curve, = ax.plot(x, x, 'r-', linewidth=2)

time_text = ax.text(.5, .5, '', fontsize=15)

#########################################

def update(i):
    #label = 'timestep {0}'.format(i)
    curve.set_ydata(predictionFull[i].data.numpy())
    time_text.set_text('Loss = %.4f' % lossFull[i].data.numpy())
    time_text.set_x(1.0)
    time_text.set_y(-3.0)
    time_text.set_color('red')

    return curve

########################################
# class

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer  # this takes (100, 1) ==> torch.Size([100, 1]) (N, n_value)
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


#########################################

net = Net(n_feature=1, n_hidden=500, n_output=1)

#optimizer = torch.optim.SGD(net.parameters(), lr=0.002)
optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

##########################################
# train and plot

ax.scatter(x.data.numpy(), y.data.numpy(), color = "orange")
ax.set_xlim(-math.pi, math.pi)
ax.set_ylim(-math.pi, math.pi)

Iterations = 2000

for t in range(Iterations):


    prediction = net(x)     # input x and predict based on x
    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()        # apply gradients
    
    predictionFull.append(prediction)
    lossFull.append(loss)
 
if __name__ == '__main__':
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, Iterations, 20), interval=2)
    anim.save('./an.gif', writer='imagemagick', fps=500)


