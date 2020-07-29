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
            * Create a new individual *i2* similar to *i1*, but with small, random differences (mutation).
            * Add *i2* to the new population.
    * Randomly select an individual (except for the one with the highest fitness) from the new population and kill it. 
    * Randomly generate a new individual and add it to the new population.
    * Discard the old population and restart the loop considering the new population.


## Trading shares of IBOVESPA
During the investment simulations, the investors are provided, for each trading day of the considered period, with historic data of the *BOVESPA Index* (or IBOVESPA), benchmark index of about 60 stocks that are traded on the B3 (Brasil Bolsa Balcão), which account for the majority of trading and market capitalization in the Brazilian stock market. The database, obtained at https://br.investing.com/, contains the opening and closing number of points the IBOVESPA had in each trading session from December 27 of 2000 to June 15 of 2020.

For the purposes of our simulations, the IBOVESPA is treated as an *exchange-traded fund* (ETF), i.e., an investment fund traded on stock exchanges, much like stocks. The price of each "IBOVESPA share" in a given instant is equal to the IBOVESPA divided by 1000.


## The investor
In each trading session, the investor (an individual of the population), represented by the class *Investor*, can decide to take one of the following actions:

   * **buy** *x* IBOVESPA shares;
   * do nothing;
   * **sell** *y* IBOVESPA shares.
   
The investor can take only one action in each trading day, during its end. In order to make a decision, the investor is provided, in each trading session, with the following informations:

   * the current amount of cash he has;
   * the number of shares he has;
   * the average price paid by the investor for his shares;
   * the closing price of the "IBOVESPA shares" for the past *d* days, where *d* is a predefined number.

The main component of each investor is its "brain". We give that name to the _**Artificial Neural Network**_ (*ANN*) responsible for making decisions for the investor. The ANN takes, as input, a vector with the information above. It outputs a real number *h*. The program rounds *h* to the nearest integer and interprets it as follows:

   * *h > 0*: action to buy *h* shares of IBOVESPA;
   * *h = 0*: no action;
   * *h < 0*: action to sell |*h*| shares of IBOVESPA.

to_do: the investor's brain; genetic algorithm + neural network = neuroevolution

to_do: actions the investor can take


## Reproduction
to_do: reward based selection vs elitism

to_do: why randomly kill one individual? The problem of a bad starting point in the search space.

to_do: a mutable mutation rate (the mutation rate gets higher when the population hasn't been changing much in the past few epochs).


## Results
to_do: using different data to train and evaluate the system

to_do: how did the investors handle crisis?

to_do: general results

<p align="center"> <img src="./Figure_1.png"width="1000" height="600"> </p> 
