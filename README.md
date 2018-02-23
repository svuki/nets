# nets

An implementation of stochastic gradient descent for multi-layer perceptrons.

## Overview of Namespaces
          
nets.net -- Contains function for constructing implementation agnostic descriptions of MLP's. Also contains functions for saving and loading MLP descriptions to files.
            
nets.sgd-handler -- Contains two main functions: sgd and sgd-handler. Both take a net description, a training-profile, a number of iterations, and an implementation interface. SGD uses implementation interface to run the computation in the current thread while sgd-handler runs the computation in another thread, allowin for limited querying and reporting of the state of the computation.

nets.implementations.* -- these are various implementation namespaces. Each implemenation is responsbile for providng and interface map with the function :net-eval, :run-sgd, :from-net, :to-vec, and :from-vec. These are used in nets.sgd-handler to run the described computation. In general an implementation will contain four parts: (1) implemnentation of activation-functions, (2) implementation of cost-functions, (3) implementation of stochastic gradient descent, and (4) the specified interface functions. Currently only the neanderthal implementation works.
                 

nets.examples.* -- various examples.

Note: as of right now only the neanderthal inmplementation works. This requires intel's math kernel library to run.

## In the Works

Provide core-matrix implementation interface.
Get GPU implementation up and running.
CNN's

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
