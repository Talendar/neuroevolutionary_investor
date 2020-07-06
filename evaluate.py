""" Auxiliary functions for displaying to the user the evaluation of a population of investors.

@Author: Gabriel Nogueira (Talendar)
"""


import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def static_plot(best_investors, ibov_var):
    """ Plots the performances of IBOVESPA and the best investors of the population.

    :param best_investors: performance history of the best investors.
    :param ibov_var: price history of IBOVESPA.
    """
    interval = range(len(ibov_var))
    ax = plt.subplot()
    plt.plot(interval, ibov_var, "g", linewidth=3, alpha=0.6)
    plt.plot(interval, best_investors[0][0], "r", linewidth=3, alpha=0.6)

    for i in best_investors[1:]:
        plt.plot(interval, i[0], linewidth=2, alpha=0.6)

    for line in ax.lines:
        y = line.get_ydata()[-1]
        ax.annotate('%0.2f%%' % y, xy=(1, y), xytext=(8, 0), color=line.get_color(),
                    xycoords=('axes fraction', 'data'), textcoords='offset points', weight="bold")

    plt.legend(['IBOV'] + ["Investor %d" % (i + 1) for i in range(len(best_investors))], loc='upper left')
    plt.xlabel("Time (days)")
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    plt.show()


def dynamic_plot(investor_history, ibov_var, print_ops):
    """ Dynamically plots, over time, the performance of IBOVESPA and on investor.

    todo: PLOTTING IS GETTING SLOWER AS MANY ANNOTATIONS ARE BEING DISPLYED; SOLVE IT!
    todo: limit the number of annotations displayed; maybe don't display annotations for every operation.

    :param investor_history: performance history of the investor.
    :param ibov_var: price history of IBOVESPA.
    :param print_ops: if True, the investor's decisions (buy and sell operations) will be plot.
    """
    # init
    plt.ion()
    figure, ax = plt.subplots()
    lines_ibov, = ax.plot([], [], "g", linewidth=3, alpha=0.6)
    lines_inv, = ax.plot([], [], "r", linewidth=3, alpha=0.6)

    BASE_PAUSE_TIME = 1
    pause_time = BASE_PAUSE_TIME

    ax.set_autoscaley_on(True)
    ax.set_xlim(0, len(ibov_var))
    ax.grid()

    plt.legend(['IBOV', "Investor"], loc='upper left')
    plt.xlabel("Time (days)")
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
    figure.subplots_adjust(left=0.25, bottom=0.25)

    # speed slider
    spd_slider_ax = figure.add_axes([0.42, 0.07, 0.3, 0.05], facecolor='lightgoldenrodyellow')
    spd_slider = plt.Slider(spd_slider_ax, 'Speed', 0.2, 20, valinit=BASE_PAUSE_TIME)

    def spd_slider_on_changed(val):
        nonlocal pause_time
        pause_time = BASE_PAUSE_TIME / val

    spd_slider.on_changed(spd_slider_on_changed)

    # plot
    xdata = []
    ydata_inv = []
    ydata_ibov = []

    pc_ann_inv = pc_ann_ibov = None
    for x in range(len(ibov_var)):
        # set data
        xdata.append(x)
        ydata_inv.append(investor_history[0][x])
        ydata_ibov.append(ibov_var[x])

        lines_inv.set_xdata(xdata)
        lines_inv.set_ydata(ydata_inv)

        lines_ibov.set_xdata(xdata)
        lines_ibov.set_ydata(ydata_ibov)

        # rescale
        ax.relim()
        ax.autoscale_view()

        # percentage annotation
        if pc_ann_ibov is not None:
            pc_ann_ibov.remove()
        pc_ann_ibov = ax.annotate('%0.2f%%' % ydata_ibov[-1], xy=(1, ydata_ibov[-1]),
                                 xytext=(8, 0), color=lines_ibov.get_color(),
                                 xycoords=('axes fraction', 'data'), textcoords='offset points', weight="bold")

        if pc_ann_inv is not None:
            pc_ann_inv.remove()
        pc_ann_inv = ax.annotate('%0.2f%%' % ydata_inv[-1], xy=(1, ydata_inv[-1]),
                                 xytext=(8, 0), color=lines_inv.get_color(),
                                 xycoords=('axes fraction', 'data'), textcoords='offset points', weight="bold")

        # op annotation
        if print_ops and investor_history[1][x][1] != 0:
            color = "g" if investor_history[1][x][0] == "BUY" else "r"
            ax.plot([xdata[-1]], [ydata_inv[-1]], marker='o', markersize=5, color=color)
            ax.annotate("%d" % investor_history[1][x][1],
                        xy=(xdata[-1], ydata_inv[-1]), xytext=(xdata[-1] - 0.25, ydata_inv[-1] - 0.25),
                        color=color, weight="bold", fontsize=8, arrowprops={"arrowstyle": "->"})

        # draw and delay
        plt.pause(pause_time)

    plt.ioff()
    plt.show()
