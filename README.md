# Master Thesis: " Neural Architecture Search: How to compromise between Time and Accuracy "

A work on the research of the architecture of Convolutional Neural Networks and how to be able to make it possible on private computer at home.

In this work, we use a cartesian genetic algorithm to design Convolutional Neural Networks based on the paper: "A Genetic Programming Approach to Designing Convolutional Neural Network Architectures" [[arXiv]](https://arxiv.org/abs/1704.00764). We combine this algorithm with a predictor to approximate the power of Convolutional Neural Networks like E2epp [[ResearchGate]](https://www.researchgate.net/publication/334008396_Surrogate-Assisted_Evolutionary_Deep_Learning_Using_an_End-to-End_Random_Forest-Based_Performance_Predictor). Furthermore, We have also implemented an equation to allow us to take into account the size of the network to prioritize a network with less layers over a network with the same accuracy but more layers. 




