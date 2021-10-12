# animatedTorchRegressor

The below code perform an animated fit of a random function using a pyTorch fully connected network. 
```
python animatedRegressor.py
```
For relatively complex function like *sin(x^2)* the network performs well for a higher number of steps. 


<img src="https://github.com/ddevetak/animatedTorchRegressor/blob/master/fun1.gif" width="600" height="400">


For a higher complex function like *cos(x^3)* the network model under-fits. 


<img src="https://github.com/ddevetak/animatedTorchRegressor/blob/master/fun2.gif" width="600" height="400">
