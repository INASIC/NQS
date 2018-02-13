# Solving the Quantum Many-Body Problem with Artificial Neural Networks

I am currently working on an undergraduate final year project that involves reproducing the results found from this paper: https://arxiv.org/abs/1606.02318. If you are interested in working with me, feel free to get in touch at ah00446@surrey.ac.uk.

The results of this paper discuss the use of a type of shallow neural network to represent the wavefunction of a quantum many body system. The network architecture used was a Restricted Boltzmann Machine (RBM), which takes as input the spins (+1 or -1) of a quantum many body system. These are provided as input to the visible layer of the RBM, and the network is trained to learn the probabiltiy distribution of these sets of spins (i.e. the wavefunction squared).

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/Restricted_Boltzmann_machine.svg/1200px-Restricted_Boltzmann_machine.svg.png)

Through the use of reinforcement learning, the ground state or unitary time evolution of the wavefunction is then determined by sampling from the RBM. Spins are sampled from the RBM using the Metropolis-Hastings sampling method and accepted if they 

### Quantum Physics Prerequisites:
* What is spin?
* The wavefunction
* Variational method
* Ising model
* Heisenberg model

### Machine Learning
* Neural Networks
* Restricted Boltzmann Machines
* Unsupervised Learning
* Metropolis-Hastings sampling method
* Reinforcement learning
