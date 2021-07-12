from Stock import *
import math
import os
import pandas as pd


data_history = 66


def write_csv(position):
    data_frame = pd.DataFrame(position).T
    if os.path.isfile("./previous_days.csv"):
        df = pd.read_csv("./previous_days.csv")
        df.loc[len(df)] = position
        os.remove("./previous_days.csv")
        df.to_csv("./previous_days.csv", index=False)
    else:
        data_frame.to_csv("./previous_days.csv", index=False)


def getMyPosition(prcSoFar):
    position = np.zeros(100)

    # CSV of previous day positions used as memory for commission-cost management
    if prcSoFar.shape[1] == 1 and os.path.isfile("./previous_days.csv"):
        os.remove("./previous_days.csv")

    # Yesterday's hold index
    previous_hold_day = (prcSoFar.shape[1] - data_history) - 1

    # Training Period
    if prcSoFar.shape[1] <= data_history:
        return position
    else:
        if prcSoFar.shape[1] == data_history + 1:
            write_csv(position)
        yesterday_positions = pd.read_csv("./previous_days.csv")

    # Holding Period
    if previous_hold_day % 2 != 0:
        prcSoFar = prcSoFar[:, :(prcSoFar.shape[1] - (previous_hold_day % 2))]

    # Data Setting Up
    stocks = []
    for i in range(100):
        vibe_periods = [15, 30, 65]
        vibe_coefficients = [4, 1, 4]
        r2_cutoffs = [0.5, 0.5, 0.8]
        new_stock = Stock(i, prcSoFar[i])
        new_stock.set_training_days(66)
        new_stock.scale_stocks()
        new_stock.update_vibe(vibe_periods, vibe_coefficients, r2_cutoffs)
        stocks.append(new_stock)

    # Method
    for index in range(100):
        investment = stocks[index].get_today_investment()
        position[index] = math.floor(investment / prcSoFar[index][-1])
        if yesterday_positions.iloc[-1][f"{index}"] != 0 and position[index] > 0:
            if (yesterday_positions.iloc[-1][f"{index}"] * prcSoFar[index][-1]) > 100000:
                position[index] = math.floor(9500 / prcSoFar[index][-1])
            else:
                position[index] = yesterday_positions.iloc[-1][f"{index}"]

    write_csv(position)
    return position
