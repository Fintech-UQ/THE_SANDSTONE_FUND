import numpy as np
from Stock import *
import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression

holding_period = 5
data_history = 24 * 5
ranking = "regression"
waiting_period = 0
std_days = 71
cut_off_max = 10
cut_off_min = 5
cut_off_r = 0.5
np.set_printoptions(suppress=True)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def set_parameters(parameters):
    global holding_period, data_history, std_days, cut_off_max, cut_off_min, cut_off_r
    (holding_period, data_history, cut_off_max, cut_off_min, cut_off_r, std_days) = parameters


def write_csv(position, anchors=np.zeros(100), threshold=np.zeros(100)):
    position = np.append(position, [anchors])
    position = np.append(position, [threshold])
    data_frame = pd.DataFrame(position).T
    if os.path.isfile("./previous_days.csv"):
        df = pd.read_csv("./previous_days.csv")
        df.loc[len(df)] = position
        os.remove("./previous_days.csv")
        df.to_csv("./previous_days.csv", index=False)
    else:
        data_frame.to_csv("./previous_days.csv", index=False)


def getMyPosition(prcSoFar, parameters, hyper):
    global holding_period, waiting_period, data_history, ranking
    set_parameters(parameters)


    position = np.zeros(100)
    if prcSoFar.shape[1] == 1 and os.path.isfile("./previous_days.csv"):
        os.remove("./previous_days.csv")

    previous_hold_day = (prcSoFar.shape[1] - data_history) - 1
    if prcSoFar.shape[1] <= data_history:
        return position
    else:
        if prcSoFar.shape[1] == data_history + 1:
            write_csv(position)

        yesterday_positions = pd.read_csv("./previous_days.csv").iloc[:, : 100]
        yesterday_anchor = pd.read_csv("./previous_days.csv").iloc[:, 100:200]
        yesterday_threshold = pd.read_csv("./previous_days.csv").iloc[:, 200:]

    if previous_hold_day % 2 != 0:
        prcSoFar = prcSoFar[:, :(prcSoFar.shape[1] - (previous_hold_day % 2))]


    # Data Setting Up
    stocks = []
    for i in range(100):
        vibe_periods = hyper[0]
        vibe_coefficients = hyper[1]
        r2_cutoffs = hyper[2]
        short_vibe_periods = hyper[3]
        short_vibe_coeficients = hyper[4]
        short_r2_cutoffs = hyper[5]

        new_stock = Stock(i, prcSoFar[i])
        new_stock.short_strat = (short_vibe_periods, short_vibe_coeficients, short_r2_cutoffs)
        new_stock.set_training_days(data_history)
        new_stock.scale_stocks()
        new_stock.update_vibe(vibe_periods, vibe_coefficients, r2_cutoffs)
        stocks.append(new_stock)

    # Method
    money_investments = []
    for i in range(100):
        money_investments.append((i, stocks[i].get_today_investment(), stocks[i].get_vibe()))

    threshold_changes = [-1, -3.5, -4.5, -5.5, -6, -8]
    anchors = np.zeros(100)
    threshold = np.zeros(100)
    for (index, investment, vibe) in money_investments:
        # if index == std_days:
        position[index] = math.floor(investment / prcSoFar[index][-1])
        # today_price = prcSoFar[index][-1]
        # previous_anchor = yesterday_anchor.iloc[-1][f"{index + 100}"]
        # if vibe > 0:
        #     if previous_anchor < today_price:
        #         anchors[index] = today_price
        #     elif previous_anchor > today_price:
        #         anchors[index] = yesterday_anchor.iloc[-1][f"{index + 100}"]
        #     position[index] = math.floor(investment / prcSoFar[index][-1])
        # else:
        #     if previous_anchor < today_price:
        #         anchors[index] = today_price
        #         position[index] = math.floor(investment / prcSoFar[index][-1])
        #     else:
        #         percent_change = (((today_price - previous_anchor) / previous_anchor) * 100)
        #         if -3 >= percent_change > -10 and yesterday_positions.iloc[-1][f"{index}"] == 0:
        #             threshold[index] = -1
        #             position[index] = math.floor(-1000 / prcSoFar[index][-1])
        #             anchors[index] = previous_anchor
        #         elif -3 >= percent_change > -10 and yesterday_positions.iloc[-1][f"{index}"] != 0:
        #             if yesterday_threshold.iloc[-1][f"{index + 200}"] <= percent_change:
        #                 threshold[index] = 0
        #                 position[index] = 0
        #                 anchors[index] = today_price
        #             else:
        #                 temp = yesterday_threshold.iloc[-1][f"{index + 200}"]
        #                 for threshold_change in threshold_changes:
        #                     if percent_change < threshold_change < yesterday_threshold.iloc[-1][f"{index + 200}"]:
        #                         temp = threshold_change
        #                     elif temp == 0:
        #                         continue
        #                     else:
        #                         break
        #                 threshold[index] = temp
        #                 position[index] = math.floor(yesterday_positions.iloc[-1][f"{index}"])
        #                 anchors[index] = yesterday_anchor.iloc[-1][f"{index + 100}"]
        #         else:
        #             anchors[index] = yesterday_anchor.iloc[-1][f"{index + 100}"]
        #             position[index] = math.floor(yesterday_positions.iloc[-1][f"{index}"])
        #             continue

    for index in range(100):
        # if index == std_days:
        if yesterday_positions.iloc[-1][f"{index}"] != 0 and position[index] > 0:
            if (yesterday_positions.iloc[-1][f"{index}"] * prcSoFar[index][-1]) > 10000:
                position[index] = 8000
            else:
                position[index] = yesterday_positions.iloc[-1][f"{index}"]

    write_csv(position, anchors, threshold)
    return position