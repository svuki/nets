# nets

An implementation of stochastic gradient descent for multi-layer perceptrons.

## Overview of Namespaces

nets.backpropgation -- an implementation of the backpropopogation 
    algorithm. Most important is the function sgd which implementes
    stochastic gradient descent. This namespace also contains 
    net-eval which can be used to evaluate a particular input with the net.
                       
nets.net -- Contains function for constructing and accesing various 
    data relate to multi layer perceptrons. Most important is is the function 
    new-net which creates a fully linked multi layer perceptron by specifying 
    its input size and the size and activation function of each of the layers.
            
nets.error-functions -- Various cost functions and their gradients.

nets.activation-functions -- various activation functions and their 
    gradient (or jacobian where appropriate).

nets.trainers -- Functions related to mediating SGD.

nets.printers -- Various procedures for printing data.

nets.matrix-utils -- Utility procedures involving matrices.

nets.utils -- Various, useful procedures.

nets.examples -- Contains four examples which compute the functions 

                 (1) x     -> 1.0 
                 
                 (2) [x y] -> [y x] 
                 
                 (3) x     -> (0.5 + sin(x)/2)
                 
                 (4) x     -> floor(x) (for x in (0, 5))
                 

In general the examples follow the following pattern, which can be used to 
train nets of your own:

(1) Make a new net using nets.net/new-net

(2) Specify a training-profilfe which is a map containg the keys 
    :input-fn :output-fn :lrate :error-function. Here :input-fn 
    corresponds to a function of 0 arity that produces a vector 
    to be fed into the net, :output-fn corresponds to a function of arity 
    1 that returns the target value of at the input produced by the 
    inptut-function, :lrate is the learning rate, and :error-function 
    corresponds to the cost function you'd like to use.
    
(3) Use core/train-for to train the net for a set number of iterations. 
    Subsequently you will see a small sample run and be given the 
    option to continue training, changing the learning rate and the number 
    of iterations should you choose to do so.

## In the Works

I'm currently working with the MNSIT handwritten digits data which is a lot
more data than the simple functions computed in the example. The backpropogation
algorithm is in the process of being optimized, and I'm working on providing a
standard framework for supervised learning to handle various data sources uniformly.

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
