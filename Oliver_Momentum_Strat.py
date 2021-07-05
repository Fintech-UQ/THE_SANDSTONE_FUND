import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler

holding_period = 5
data_history = 24 * 5
ranking = "growth"
waiting_period = 6
no_stocks = 15
std_days = 60
np.set_printoptions(suppress=True)


def get_current_hold_prices(prcSoFar):
    tmp = prcSoFar.shape[1] % holding_period - 1
    days_since_hold = holding_period - 1 if tmp == -1 else tmp
    current_index = prcSoFar.shape[1] - 1
    monday_index = current_index - days_since_hold
    return prcSoFar.T[:monday_index].T


def get_volatility(data, days):
    results = []
    for i in range(100):
        std = np.std(data[i][-days:]) * np.sqrt(days)
        results.append(std)
    return results


def get_returns(prices_up_until_monday):
    stocks = []
    for i, stock in enumerate(prices_up_until_monday):
        index = len(stock) - 1
        start_monday_index = index - data_history
        end_friday_index = index - waiting_period
        percent_return = ((stock[end_friday_index] - stock[start_monday_index]) / stock[end_friday_index]) * 100
        stocks.append(percent_return)
    return stocks


def update_position(currentPositions, prcSoFar):
    for i, position in enumerate(currentPositions):
        if position * prcSoFar[i][-1] > 10000:
            currentPositions[i] = math.floor(5000 / prcSoFar[i][-1])
            print(f"Sold {i} on day {prcSoFar.shape[1]}")


def get_data(prcSoFar):
    returns = get_returns(prcSoFar)
    std = get_volatility(prcSoFar, std_days)
    results = []
    for i in range(100):
        results.append((i, returns[i], std[i]))
    return results


def set_parameters(parameters):
    global holding_period, waiting_period, data_history, ranking, no_stocks, std_days
    (holding_period, data_history, ranking, waiting_period, no_stocks, std_days) = parameters


def scale(prHst):
    stocks = []
    for old_stock in prHst:
        stock = []
        start_price = old_stock[0]
        for price in old_stock:
            stock.append(price/start_price)
        stocks.append(stock)
    return stocks


def get_position_size(hold, short):
    hold.sort(key=lambda x: x[1], reverse=True)
    short.sort(key=lambda x: x[1])
    benchmark_hold = hold[0][1]
    benchmark_short = short[0][1]

    positions = []
    for (index, value) in hold:
        positions.append((index, (value/benchmark_hold) * 9500))

    for (index, value) in short:
        positions.append((index, -1 * (value/benchmark_short) * 9500))

    return positions



def getMyPosition(prcSoFar, parameters):
    global holding_period, waiting_period, data_history, ranking, no_stocks
    set_parameters(parameters)
    position = np.zeros(100)

    if prcSoFar.shape[1] <= data_history:
        return np.zeros(100)

    current_hold_prices = get_current_hold_prices(prcSoFar)
    scaled_data = scale(current_hold_prices)
    stock_data = get_data(scaled_data)

    if ranking == "growth":
        stock_data.sort(key=lambda x: x[1], reverse=True)
        winners = stock_data[:no_stocks]
        for i, winner in enumerate(winners):
            index, ret, std = winner
            price = current_hold_prices[index][-1]
            number_of_shares = math.floor((5000/(i + 1)) / price)
            if ret > 0:
                position[index] = number_of_shares
    elif ranking == "hybrid":
        array = np.array(stock_data)
        df = pd.DataFrame(array, columns=("index", "returns", "std"))
        x = np.linspace(0, 1, 1000)
        y = 1 + 0.25 * x
        y_neg = -1 * y
        y_zero = np.zeros(1000)

        hold = []
        short = []
        dead_zone = []
        for i, (x_value, y_value) in enumerate(zip(df["std"], df["returns"])):
            expected_y_positive = 1 + 0.56 * x_value
            expected_y_negative = -1 * expected_y_positive
            if y_value >= expected_y_positive:
                hold.append((i,(y_value - 1) / x_value))
            elif y_value <= expected_y_negative:
                short.append((i, (y_value - 1) / x_value))
            else:
                dead_zone.append((i, y_value / x_value))

        positions = get_position_size(hold, short)

        for (index, money) in positions:
            if money > 0:
                position[index] = math.floor(money/current_hold_prices[index][-1])

        # fig, ax = plt.subplots()
        # ax.scatter(df["std"], df["returns"])
        # ax.scatter(x, y, marker='_')
        # for i, (x_1, y_1) in enumerate(zip(df["std"], df["returns"])):
        #     ax.annotate(i, (x_1, y_1))
        # ax.scatter(x, y_neg, marker='_')
        # ax.scatter(x, y_zero, marker='_')
        # plt.show()

    # update_position(position, prcSoFar)

    return position