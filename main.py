""" Neuroevolutionary Investor

Work in progress.
@Author Gabriel Nogueira (Talendar)
"""


from investor import InvestorPopulation
import pandas as pd
from pathlib import Path
from datetime import datetime
import os


def format_csv():
    ibov = pd.read_csv("./data/ibov_dados_originais.csv", index_col=None, usecols=["Data", "Último", "Abertura"])
    ibov = ibov[::-1].reset_index(drop=True)

    for i, row in ibov.iterrows():
        row["Último"] = row["Último"].replace(".", "").replace(",", ".")
        row["Abertura"] = row["Abertura"].replace(".", "").replace(",", ".")

    ibov.to_csv("./data/ibov_dados.csv", index=False)


def train():
    """ Handles the training of the investors (menu option 0). """
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
    best = pop.evolve(generations=gens, prices=ibov["Último"][start_day:end_day].values / 1000)

    # saving results
    out_dir = "./out/" + f"{datetime.now():%y-%m-%d-%H-%M-%S}" + "_pop%d/" % pop_size
    if not os.path.isdir(out_dir):
        Path(out_dir).mkdir()

    pop.save(out_dir)
    print("\nResults saved to: %s" % out_dir)


def evaluate():
    """ Handles the evaluation of a trained population (menu option 1). """
    # input
    print("\n\nInput file (csv at ./data): ", end="")
    in_file = input()
    ibov = pd.read_csv("./data/" + in_file, index_col=None, usecols=["Data", "Último", "Abertura"])

    print("Population directory path (e.g: ./out/my_pop/): ", end="")
    in_dir = input()
    pop = InvestorPopulation(in_dir=in_dir)

    print("Considered days (start, end): ", end="")
    start_day, end_day = [int(x) for x in input().split(" ")]

    print("Number of investors to plot (min 1, max %d): " % pop.size(), end="")
    num_plot = int(input())

    if num_plot <= 0 or num_plot > pop.size():
        print("Error: invalid population size!")
        return

    # evaluating
    print("\nStarting date (day 0): %s  |  Pts: %.2f" % (ibov["Data"].iloc[pop._prev_days], ibov["Último"].iloc[pop._prev_days]))
    print("Ending date (day %d): %s  |  Pts: %.2f" % (end_day - 1 - pop._prev_days, ibov["Data"].iloc[end_day - 1], ibov["Último"].iloc[end_day - 1]))
    print("Evaluating... ", end="")
    pop.evaluate(ibov["Último"][start_day:end_day].values / 1000, num_plot)
    print("done!")


if __name__ == "__main__":
    opt = -1
    while opt != 2:
        print(
            "\n\n< Nuero-Evolutionary Investor (by Talendar) >\n"
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





