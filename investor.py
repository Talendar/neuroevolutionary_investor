""" This module contains the classes that represent investors and populations of investors.

@Author: Gabriel Nogueira (Talendar)
"""


import os
import tensorflow as tf
import numpy as np
import numpy.random as rand
import time


################## CONFIG ##################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # turn off TensorFlow's logs
MIN_MUTATION_RATE = 0.05                   # minimum mutation rate for the genetic algorithm
MAX_MUTATION_RATE = 0.5                    # maximum mutation rate for the genetic algorithm
WEIGHTS_MULT_FACTOR = 200                  # factor that multiplies the weights of a newly created neural network
############################################


class Investor:
    """ Defines the general structure and behaviour of an automated intelligent investor.

    Attributes:
        _cash: the investor's current amount of free cash.
        _stocks: number of stocks the investor currently has.
        _avg_price: the average price the investor paid for the stocks.
        brain: the neural network that dictates the investor's actions.
    """

    def __init__(self, initial_cash, prev_days, weights=None):
        """ Standard constructor for an instance of Investor.

        :param initial_cash: initial amount of cash the investor will have available.
        :param prev_days: number of previous days of stock prices the investor will take into account.
        :param weights: initial weights of the investor's neural network. If None, new random weights are created.
        """
        self._cash = initial_cash
        self._stocks = self._avg_price = 0
        self.brain = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=[4 + prev_days]),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation="linear")
        ])

        self._multiply_weights()
        if weights is not None:
            self.brain.set_weights(weights)

    def _multiply_weights(self):
        """ Multiplies the network's weights by a given factor. """
        self.brain.set_weights([w * WEIGHTS_MULT_FACTOR for w in self.brain.get_weights()])

    def fitness(self, price):
        """ Returns the current net worth of the investor. """
        return self._cash + self._stocks * price

    def operate(self, features):
        """ Simulates investments using the given data. Returns the name of the action taken by the investor. """
        features[0][0] = self._cash
        features[0][1] = self._stocks
        features[0][2] = self._avg_price

        price_now = features[0][3]
        h = round(self.brain.predict(features)[0][0])

        # buy
        if h > 0:
            h = int(self._cash / price_now) if h > int(self._cash / price_now) else h
            self._avg_price = (self._avg_price * self._stocks + h * price_now) / (self._stocks + h)
            self._stocks += h
            self._cash -= h * price_now
            return "BUY", h

        # sell
        if h < 0:
            h = self._stocks if abs(h) > self._stocks else abs(h)
            self._stocks -= h
            self._avg_price = 0 if self._stocks == 0 else self._avg_price
            self._cash += h * price_now
            return "SELL", h

        # no action
        return "NA", 0

    def reset_net_worth(self, initial_cash):
        """ Resets the investor's net worth and sets its cash to the specified amount. """
        self._cash = initial_cash
        self._stocks = self._avg_price = 0

    def save_brain(self, out_pathname):
        """ Saves the investor's neural network to disk. """
        self.brain.save(out_pathname)

    def load_brain(self, in_pathname):
        """ Loads the investor's neural network from disk. """
        self.brain = tf.keras.models.load_model(in_pathname, compile=False)


class InvestorPopulation:
    """ Represents a population of investors.

    This class is used to manage a population of investors, dealing with fitness evaluation, reproduction, etc.

    Attributes:
        _initial_cash: initial amount of cash of each of the population's member.
        _prev_days: number of previous days of stock prices the investors will take into account.
        _investors: a list with Investor objects.
    """

    def __init__(self, in_dir=None, pop_size=None, initial_cash=None, prev_days=None):
        """ Standard constructor for a population of investors.

        :param in_dir: input directory from which the population will be loaded. If None, a new population is generated.
        :param pop_size: size of the population.
        :param initial_cash: initial amount of cash each investor will have at the beginning of a new generation.
        :param prev_days: number of previous days of stock prices the investors will take into account.
        """
        self._fitness_history = []

        # create from file
        if in_dir is not None:
            with open(in_dir + "info.txt", "r") as file:
                self._initial_cash = float(file.readline())
                pop_size = int(file.readline())
                self._prev_days = int(file.readline())

            self._investors = [Investor(initial_cash, self._prev_days) for _ in range(pop_size)]
            for count, inv in enumerate(self._investors):
                inv.load_brain(in_dir + "brain%d.h5" % count)

        # create from args
        else:
            self._initial_cash = initial_cash
            self._prev_days = prev_days
            self._investors = [Investor(initial_cash, prev_days) for _ in range(pop_size)]

    def size(self):
        """ Returns the size of the population. """
        return len(self._investors)

    @property
    def prev_days(self):
        """ Returns the number of previous days of stock prices the investors are taking into account."""
        return self._prev_days

    def _mutation_rate(self):
        """ Calculates the mutation rate.

        The mutation rate is higher when the population hasn't been improving its fitness much in the past few
        generations and lower when the population has been improving.
        """
        m = np.mean( self._fitness_history[-6:-1] if len(self._fitness_history) > 5 else self._fitness_history )
        delta = self._fitness_history[-1] - m

        sig = 1 / (1 + np.exp(delta))
        return max(MIN_MUTATION_RATE, MAX_MUTATION_RATE * sig)

    def evolve(self, generations, prices):
        """ Starts the evolutionary process.

        Uses a genetic algorithm. By the end of the process, the population fitness is expected to have been improved.

        :param generations: number of generations of the process.
        :param prices: stock price history.
        :return: the best individual.
        """
        features = _generate_features(prices, self._prev_days)
        fitness = None
        initial_price, final_price = features[0][0][3], features[-1][0][3]

        elapsed_time = 0
        for g in range(generations):
            start_time = time.time()
            print(
                "\n\n< GENERATION %d/%d\n" % (g+1, generations) +
                "  Investing... ", end=""
            )

            # resetting investors
            for investor in self._investors:
                investor.reset_net_worth(self._initial_cash)

            # simulating investments
            for f in features:
                for investor in self._investors:
                    investor.operate(f)
            print("done!")

            # calculating fitness
            print("  Calculating profits... ", end="")
            fitness = sorted([(n, i.fitness(final_price)) for n, i in enumerate(self._investors)], key=lambda e: e[1])

            best_profit = fitness[-1][1] - self._initial_cash
            print("done! Best profit: %.2f (%.2f%%)" % (best_profit, 100*best_profit/self._initial_cash))
            print("  Current population fitness: " + str([f[1] for f in fitness]))

            f_mean = np.mean([f[1] for f in fitness])
            print("  Mean population fitness: %.2f" % f_mean)
            self._fitness_history.append(f_mean)

            market_profit = final_price - initial_price
            print("  Market variation: %.2f (%.2f%%)" % (market_profit, 100*market_profit/initial_price))

            # reproducing
            print("  Mutation rate: %.2f%%" % (100*self._mutation_rate()))
            print("  Reproducing... ", end="")
            self._reward_based_reproduction(fitness)
            self._random_death()
            print("done! >")

            # ETA
            elapsed_time += time.time() - start_time
            remaining_time = elapsed_time * (generations - g + 1) / (g + 1)
            m, sec = divmod(remaining_time, 60)
            hour, m = divmod(m, 60)
            print("\nETA: %02dh %02dmin %02ds" % (hour, m, sec))

        return self._investors[fitness[-1][0]]  # returning the best investor

    def _reward_based_reproduction(self, fitness):
        """ Reproduction method: reward-based selection. """
        p = np.array([2**i for i in range(len(fitness))])
        p = p/p.sum()    # probabilistic distribution
        new_investors = [ self._investors[ fitness[-1][0] ] ]   # always keeps the best individual

        for _ in range(len(fitness) - 1):
            choice = fitness[np.random.choice(len(fitness), p=p)]
            parent_w = self._investors[choice[0]].brain.get_weights()
            new_investors.append(Investor(self._initial_cash, self._prev_days,
                                          weights=mutate_weights(parent_w, self._mutation_rate())))

        self._investors = new_investors

    def _elitist_reproduction(self, fitness):
        """ Reproduction method: elitism. """
        best = self._investors[fitness[-1][0]]
        new_investors = [best]  # always keeps the best individual

        for i in fitness[:-1]:
            current_inv = self._investors[i[0]]
            new_weights = mate_weights(best.brain.get_weights(), current_inv.brain.get_weights(), self._mutation_rate())
            new_investors.append(Investor(self._initial_cash, self._prev_days, weights=new_weights))

    def _random_death(self):
        """ Randomly kills one of the population's individuals, replacing it with a randomly generated one.

        The best individual won't be considered for removal. It must be located at the index 0 of "self._investors".
        """
        i = np.random.randint(1, len(self._investors))
        del self._investors[i]
        self._investors.append(Investor(self._initial_cash, self._prev_days))

    def evaluate(self, prices):
        """ Evaluates the population performance.

        :param prices: stock price history.
        :return: a tuple containing, respectively, a list (sorted from best to worst) with the population's investors
        performance over time and a list with the IBOVESPA performance over the time.
        """
        features = _generate_features(prices, self._prev_days)
        initial_price = features[0][0][3]

        # resetting investors
        for investor in self._investors:
            investor.reset_net_worth(self._initial_cash)

        # simulating investments
        ibov_var = []
        investors_info = [([], []) for _ in range(len(self._investors))]   # index 0: net worth var; index 1: actions

        for f in features:
            current_price = f[0][3]
            ibov_var.append(100 * (current_price - initial_price) / initial_price)
            for n, investor in enumerate(self._investors):
                investors_info[n][1].append( investor.operate(f) )
                investors_info[n][0].append(
                        100 * ( investor.fitness(current_price) - self._initial_cash ) / self._initial_cash
                )

        return sorted(investors_info, key=lambda e: e[0][-1], reverse=True), ibov_var

    def save(self, out_dir):
        """ Saves the entire population to the given pathname. """
        with open(out_dir + "info.txt", "w") as file:
            file.write("%f\n" % self._initial_cash)
            file.write("%d\n" % len(self._investors))
            file.write("%d\n" % self._prev_days)

        for count, i in enumerate(self._investors):
            i.save_brain(out_dir + "brain%d.h5" % count)


def mutate_weights(weights, rate):
    """ Returns a mutated copy of the given set of weights.

    :param weights: the weights of a neural network.
    :param rate: the mutation rate.
    :return: a mutated copy of the set of weights.
    """
    mul = [rand.uniform(low=(1-rate), high=(1+rate), size=w.shape) for w in weights]
    return [a * b for a, b in zip(weights, mul)]


def mate_weights(weights1, weights2, mutation_rate):
    """ Sums each weight of one set with the corresponding weight of the other set. The result is divided by 2 and the
    mutation rate is applied.

    :param weights1: the first set of weights.
    :param weights2: the second set of weights.
    :param mutation_rate: the mutation rate.
    :return: the resultant set of weights.
    """
    n = [(w1 + w2)/2 for w1, w2 in zip(weights1, weights2)]
    return mutate_weights(n, mutation_rate)


def _generate_features(prices, prev_days):
    """ Generates a set of features from the given stock price history. """
    features = []
    day = prev_days
    for p in prices[prev_days:]:
        f = np.zeros(4 + prev_days)
        f[3] = p
        f[4:] = prices[(day - prev_days):day]
        f.shape = (1, len(f))
        features.append(f)
        day += 1
    return features
