""" This module contains the classes that represent investors and populations of investors.

@Author: Gabriel Nogueira (Talendar)
"""


import os
import tensorflow as tf
import numpy as np
import numpy.random as rand
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
MIN_MUTATION_RATE = 0.05
MAX_MUTATION_RATE = 0.5
WEIGHTS_MULT_FACTOR = 200


class Investor:
    """ Represents an investor, defining its structure and behaviour.

    Attributes:
        _cash: todo
        _stocks: todo
        _avg_price: todo
        _prev_days: todo
        brain: todo
    """

    def __init__(self, initial_cash, prev_days, weights=None):
        """ Standard constructor for an instance of Investor.

        :param initial_cash: initial amount of cash the investor will have available.
        :param prev_days:
        :param weights:
        """
        self._cash = initial_cash
        self._stocks = self._avg_price = 0
        self._prev_days = prev_days
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
        """ Returns the current net worth of the investor.
        """
        return self._cash + self._stocks * price

    def operate(self, features):
        """ Simulates investments using the given data.

        :param features:
        :return:
        """
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
        # sell
        elif h < 0:
            h = self._stocks if abs(h) > self._stocks else abs(h)
            self._stocks -= h
            self._avg_price = 0 if self._stocks == 0 else self._avg_price
            self._cash += h * price_now

    def reset_net_worth(self, initial_cash):
        """

        :return:
        """
        self._cash = initial_cash
        self._stocks = self._avg_price = 0

    def save_brain(self, out_pathname):
        """

        :param out_pathname:
        :return:
        """
        self.brain.save(out_pathname)

    def load_brain(self, in_pathname):
        """

        :param in_pathname:
        :return:
        """
        self.brain = tf.keras.models.load_model(in_pathname, compile=False)


class InvestorPopulation:
    """ Represents a population of investors.

    This class is used to manage a population of investors, dealing with fitness evaluation, reproduction, etc.

    Attributes:
        _initial_cash: todo
    """

    def __init__(self, in_dir=None, pop_size=None, initial_cash=None, prev_days=None):
        """ Standard constructor for a population of investors.

        :param in_dir:
        :param pop_size: size of the population.
        :param initial_cash: the amount of cash each investor will have at the beginning of a new generation.
        :param prev_days:
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

    def _mutation_rate(self):
        """ Calculates the mutation rate. """
        m = np.mean( self._fitness_history[-6:-1] if len(self._fitness_history) > 5 else self._fitness_history )
        delta = self._fitness_history[-1] - m

        sig = 1 / (1 + np.exp(delta))
        return max(MIN_MUTATION_RATE, MAX_MUTATION_RATE * sig)

    def evolve(self, generations, prices):
        """ Starts the simulation.

        :param generations:
        :param prices:
        :return:
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
            self.reproduction1(fitness)
            print("done! >")

            # ETA
            elapsed_time += time.time() - start_time
            remaining_time = elapsed_time * (generations - g + 1) / (g + 1)
            m, sec = divmod(remaining_time, 60)
            hour, m = divmod(m, 60)
            print("\nETA: %02dh %02dmin %02ds" % (hour, m, sec))

        return self._investors[fitness[-1][0]]  # returning the best investor

    def reproduction1(self, fitness):
        """ Reward-based selection. """
        p = np.array([2**i for i in range(len(fitness))])
        p = p/p.sum()    # probabilistic distribution
        print("\nProb. distribution: " + str(p))
        new_investors = [self._investors[ fitness[-1][0] ]]   # always keeps the best individual

        for _ in range(len(fitness) - 1):
            choice = fitness[np.random.choice(len(fitness), p=p)]
            parent_w = self._investors[choice[0]].brain.get_weights()
            new_investors.append(Investor(self._initial_cash, self._prev_days, weights=mutate_weights(parent_w, self._mutation_rate())))

        self._investors = new_investors

    def reproduction2(self, fitness):
        """ Elitism. """
        best = self._investors[fitness[-1][0]]
        new_investors = [best]  # always keeps the best individual

        for i in fitness[:-1]:
            current_inv = self._investors[i[0]]
            new_weights = mate_weights(best.brain.get_weights(), current_inv.brain.get_weights(), self._mutation_rate())
            new_investors.append(Investor(self._initial_cash, self._prev_days, weights=new_weights))

    def evaluate(self, prices, num_plots):
        """ Evaluates the population performance. """
        features = _generate_features(prices, self._prev_days)
        initial_price = features[0][0][3]

        # resetting investors
        for investor in self._investors:
            investor.reset_net_worth(self._initial_cash)

        # simulating investments
        ibov_var = []
        investors_var = [[] for _ in range(len(self._investors))]

        for f in features:
            current_price = f[0][3]
            ibov_var.append(100 * (current_price - initial_price) / initial_price)
            for n, investor in enumerate(self._investors):
                investor.operate(f)
                investors_var[n].append( 100*(investor.fitness(current_price) - self._initial_cash) / self._initial_cash )

        # select the n best investors
        best_var = sorted(investors_var, key=lambda e: e[-1], reverse=True)[:num_plots]

        # plot
        interval = range(len(features))
        ax = plt.subplot()
        plt.plot(interval, ibov_var, "y")
        plt.plot(interval, best_var[0], "r")

        for i in best_var[1:]:
            plt.plot(interval, i)

        for line in ax.lines:
            y = line.get_ydata()[-1]
            ax.annotate('%0.2f%%' % y, xy=(1, y), xytext=(8, 0), color=line.get_color(),
                        xycoords=('axes fraction', 'data'), textcoords='offset points', weight="bold")

        plt.legend(['IBOV'] + ["Investor %d" % (i+1) for i in range(len(best_var))], loc='upper left')
        plt.xlabel("Time (days)")
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
        plt.show()

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

    :param weights:
    :param rate:
    :return:
    """
    mul = [rand.uniform(low=(1-rate), high=(1+rate), size=w.shape) for w in weights]
    return [a * b for a, b in zip(weights, mul)]


def mate_weights(weights1, weights2, mutation_rate):
    """

    :param weights1:
    :param weights2:
    :param mutation_rate:
    :return:
    """
    n = [(w1 + w2)/2 for w1, w2 in zip(weights1, weights2)]
    return mutate_weights(n, mutation_rate)


def _generate_features(prices, prev_days):
    """ Generates a set of features. """
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
