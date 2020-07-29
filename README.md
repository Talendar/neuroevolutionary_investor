# Neuroevolutionary Investor
Neuroevolution-based system that simulates investments in the stock market, based on the BOVESPA index. Final project of the course _SSC0713 - Evolutionary systems applyied to robotics_.


## Installation
First, make sure you have a recent version of Python 3. To install the dependencies, open a command shell in the project's root directory and execute the following:
- pip install -r requirements.txt

To run the program, use the command:
- python3 main.py


## How it works
To improve itself, the system uses concepts of **evolutionary computing**. Specifically, a **genetic algorithm**, metaheuristic inspired by the process of natural selection, is used. The evolution process used by the program can be summarized in the following steps:

* Generate a population of *n* individuals; each individual represents an investor.
* Repeat *e* times, where *e* is the number of *epochs* considered:
    * For each investor present in the population, do:
        * Simulate the investor's actions (buying and selling shares) in the stock exchange over a predefined period of time.
        * Calculate and save the investor's *fitness*, defined as the percentage of profit it made during the simulation.
    * Generate a new population of *n* individuals, following the steps:
        * Add the individual with the highest fitness (investor with the best profit) to the new population.
        * Repeat *n - 1* times:
            * Consider a probabilistic distribution in which individuals with higher fitness are more likely to be selected.
            * Select one individual *i1* of the old population (without removing it).
            * Create a new individual *i2* similar to *i1*, but with small differences (mutation).
            * Add *i2* to the new population.
    * Randomly select an individual (except for the one with the highest fitness) of the new population and kill it. 
    * Randomly generate a new individual and add it to the new population.
    * Discard the old population and restart the loop considering the new population.

## Results
to do

<p align="center"> <img src="./Figure_1.png"width="1000" height="600"> </p> 
