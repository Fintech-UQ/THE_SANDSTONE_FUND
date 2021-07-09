import numpy as np
from Stock import *
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression

holding_period = 5
data_history = 24 * 5
ranking = "regression"
waiting_period = 0
std_days = 60
cut_off_max = 10
cut_off_min = 5
cut_off_r = 0.5
np.set_printoptions(suppress=True)


def set_parameters(parameters):
    global holding_period, data_history, std_days, cut_off_max, cut_off_min, cut_off_r
    (holding_period, data_history, cut_off_max, cut_off_min, cut_off_r, std_days) = parameters


def getMyPosition(prcSoFar, parameters, hyper):
    global holding_period, waiting_period, data_history, ranking
    set_parameters(parameters)
    position = np.zeros(100)

    yesterday_positions = []
    previous_hold_day = (prcSoFar.shape[1] - data_history) - 1
    yesterday_prices = []
    if prcSoFar.shape[1] <= data_history:
        return position
    elif previous_hold_day % 3 != 0:
        prcSoFar = prcSoFar[:, :(prcSoFar.shape[1] - (previous_hold_day % 3))]


    # Data Setting Up
    stocks = []
    for i in range(100):
        vibe_periods = hyper[0]
        vibe_coefficients = hyper[1]
        r2_cutoffs = hyper[2]
        new_stock = Stock(i, prcSoFar[i])
        new_stock.set_training_days(data_history)
        new_stock.scale_stocks()
        new_stock.update_vibe(vibe_periods, vibe_coefficients, r2_cutoffs)
        stocks.append(new_stock)

    # Method
    money_investments = []
    for i in range(100):
        money_investments.append((i, stocks[i].get_today_investment(), stocks[i].get_vibe()))

    for (index, investment, vibe) in money_investments:
        position[index] = math.floor(investment / prcSoFar[index][-1])


    return position