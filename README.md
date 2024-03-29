# Numerical-Simulations

This repository contains code that I originally wrote in https://github.com/parleyyang/OPRG1. For the purposes of seperating it out from things I didn't write and since that other repository contains other less relevant things, this repository contains my code base for implementing a set of numerical simulations.

The code was written for this paper https://arxiv.org/abs/2110.11156v2, and closely matches the methodology you'll find in section 3 (simulations) of the paper.

The codebase contains a set of artificial numerical simulations that are related to a phenomenon observed in time series, called regime-switching. Regime-switching occurs when the properties of the data change part way through the dataset, i.e. when we "switch" from one regime to another. A simple example is a recessionary period versus an expansionary period, say, in a macroeconomic dataset, where clearly the properties of the data are state-dependent.

In reality, regime-switching is much more subtle, and rather than switching between binary states the distinctions are much more blurred. This makes studying the behaviour of models under regime-switching difficult, since there is always a question of which state you are actually in. There is actually an entire field of study called change point detection devoted entirely to this question.

Luckily, simulations allow us to create own our idealised experiments where we control how the system behaves and can conduct this analysis in a much more controlled setting. Rather than being bombarded by noise and trying to figure out which regime we are in, we specify the regimes ourselves. We can study how our models behave before and after the regime-switch and perhaps that can tell us about how they might behave under the hazier conditions we find empirically. 

Enough of the philosopy of simulations though! Here is an outline to the code base:

**System A**: System A is a univariate system, where we simply simulate one-dimensional time series with autoregressive dependence. Half way through the simulated     dataset, the process switches from an AR(1) with a mean of 2 and an autoregressive coefficient of 0.1 to an AR(2) with a mean of 8 and autoregressive              coefficients of 0.2 and -0.5 respectively. 

**System B**: System B is a multivariate system, where we simulate various multivariate time series which take the form of [VAR systems](https://en.wikipedia.org/wiki/Vector_autoregression). Once again, the properties of the system change part way through the dataset. In this case, however, the change occurs both in the autoregressive behaviour (how each time series relates to itself), but also in the interactions between variables.

**System C**: System C is a stochastic process where we simulate a random walk to mimic a stock price. In this case, we simulate an actual financial market crash to occur at either t = 50 or t = 70 in the dataset with uniform probability, effectively partitioning the dataset into "boom" and "bust" periods.

Examples of variables simulated from each of the 3 systems are displayed below. For system A (left) and system B (middle), the vertical red line marks the regime-switch. For system C, the blue and orange lines mark the two possible points that a regime-switch can occur. Overall, we ran 10000 iterations for each system. For more details, do have a look at the results in the paper!


![fig1](https://user-images.githubusercontent.com/55145311/147833359-95068b7b-aee2-4782-8f71-4c6ac9ac6577.png)
