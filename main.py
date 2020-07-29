""" Neuroevolutionary Investor

Work in progress.
@Author Gabriel Nogueira (Talendar)
"""


import os
from datetime import datetime
from pathlib import Path
import pandas as pd
from investor import InvestorPopulation
from evaluate import *


def format_csv():
    """ Auxiliary function used to format the raw IBOVESPA data. """
    ibov = pd.read_csv("./data/ibov_dados_originais.csv", index_col=None, usecols=["Data", "Último", "Abertura"])
    ibov = ibov[::-1].reset_index(drop=True)

    for i, row in ibov.iterrows():
        row["Último"] = row["Último"].replace(".", "").replace(",", ".")
        row["Abertura"] = row["Abertura"].replace(".", "").replace(",", ".")

    ibov.to_csv("./data/ibov_dados.csv", index=False)


def train():
    """ Handles the training of an investor population (menu option 0). """

    # input
    print("\n\nInput file (csv at ./data): ", end="")
    in_file = input()
    ibov = pd.read_csv("./data/" + in_file, index_col=None, usecols=["Data", "Último", "Abertura"])

    print("Training days (start, end): ", end="")
    start_day, end_day = [int(x) for x in input().split(" ")]

    print("Population size: ", end="")
    pop_size = int(input())

    print("Initial cash: ", end="")
    initial_cash = float(input())

    print("Number of previous days to consider: ", end="")
    prev_days = int(input())

    print("Number of training generations: ", end="")
    gens = int(input())

    # training
    pop = InvestorPopulation(pop_size=pop_size, initial_cash=initial_cash, prev_days=prev_days)
    pop.evolve(generations=gens, prices=ibov["Último"][start_day:end_day].values / 1000)

    # saving results
    out_dir = "./out/" + f"{datetime.now():%y-%m-%d-%H-%M-%S}" + "_day%dto%d/" % (start_day, end_day)
    if not os.path.isdir(out_dir):
        Path(out_dir).mkdir()

    pop.save(out_dir)
    print("\nResults saved to: %s" % out_dir)


def evaluate():
    """ Handles the evaluation of a trained investors population (menu option 1). """

    # base input
    print("\n\nIBOVESPA data file (csv at ./data): ", end="")
    in_file = input()
    ibov = pd.read_csv("./data/" + in_file, index_col=None, usecols=["Data", "Último", "Abertura"])

    print("Population directory path (e.g: ./out/my_pop/): ", end="")
    in_dir = input()
    pop = InvestorPopulation(in_dir=in_dir)

    print("Considered days (start, end): ", end="")
    start_day, end_day = [int(x) for x in input().split(" ")]

    if (end_day - start_day) < (pop.prev_days + 1):
        print("Not enough days! This population needs at least %d previous days of data." % pop.prev_days)
        return

    # evaluating
    print("\nStarting date (day 0): %s  |  Pts: %.2f" %
          (ibov["Data"].iloc[start_day + pop.prev_days], ibov["Último"].iloc[start_day + pop.prev_days]))
    print("Ending date (day %d): %s  |  Pts: %.2f" % (end_day - start_day - 1, ibov["Data"].iloc[end_day - 1],
                                                      ibov["Último"].iloc[end_day - 1]))
    print("Evaluating... ", end="")
    investors_history, ibov_var = pop.evaluate(ibov["Último"][start_day:end_day].values / 1000)
    print("done!")

    # plotting
    plt_opt = -1
    while plt_opt != 2:
        print(
            "\n\n> Plot Population Evolution\n"
            "   [0] Static plot\n"
            "   [1] Dynamic plot\n"
            "   [2] Leave\n"
            "Option: ", end=""
        )
        plt_opt = int(input())

        # static plot
        if plt_opt == 0:
            print("\nNumber of investors to plot (min 1, max %d): " % pop.size(), end="")
            num_plot = int(input())

            if num_plot <= 0 or num_plot > pop.size():
                print("Error: invalid population size!")
            else:
                static_plot(investors_history[:num_plot], ibov_var)
        # dynamic plot
        elif plt_opt == 1:
            print("\nInvestor to plot (%d is the best and %d is the worst): " % (1, pop.size()), end="")
            inv_num = int(input())

            print("Plot investor's decisions (y or n): ", end="")
            print_ops = (input() == 'y')

            if print_ops:
                print("\nThe investor's decisions will be plotted. A green number means a \"buy\" operation and a red "
                      "number means a \"sell\" operation.\n"
                      "The decisions are displayed only when 1 or more stocks are bought or sold.\n"
                      "Over larger periods, it may be difficult to see the operations on the plot. Use the zoom to "
                      "better visualize them in these cases.")

            if inv_num < 1 or inv_num > pop.size():
                print("Invalid investor!")
            else:
                dynamic_plot(investors_history[inv_num - 1], ibov_var, print_ops)


if __name__ == "__main__":
    opt = -1
    while opt != 2:
        print(
            "\n\n< Neuro-Evolutionary Investor (by Talendar) >\n"
            "   [0] Train\n"
            "   [1] Evaluate\n"
            "   [2] Exit\n"
            "Option: ", end=""
        )
        opt = int(input())

        # train
        if opt == 0:
            train()
        # evaluate
        elif opt == 1:
            evaluate()

    print("\nLeaving...\n")
