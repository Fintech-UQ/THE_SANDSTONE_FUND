import numpy as np
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
    x_values = np.array(list(range(prcSoFar.shape[1] - std_days, prcSoFar.shape[1]))).reshape(-1, 1)
    results = []
    for i in range(100):
        reg_model = LinearRegression()
        reg_model.fit(x_values, prcSoFar[i][-std_days:])
        results.append((i, returns[i], std[i], reg_model.score(x_values, prcSoFar[i][-std_days:])))
    return results


def set_parameters(parameters):
    global holding_period, data_history, std_days, cut_off_max, cut_off_min, cut_off_r
    (holding_period, data_history, cut_off_max, cut_off_min, cut_off_r, std_days) = parameters


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


def ratio_function(ret):
    ret = abs(ret)
    if ret > cut_off_max:
        return 1
    elif ret > cut_off_min:
        return ret/cut_off_min - 1
    else:
        return 0


def getMyPosition(prcSoFar, parameters):
    global holding_period, waiting_period, data_history, ranking
    set_parameters(parameters)
    position = np.zeros(100)

    if prcSoFar.shape[1] <= data_history:
        return position

    current_hold_prices = get_current_hold_prices(prcSoFar)
    scaled_data = scale(current_hold_prices)
    stock_data = get_data(np.array(scaled_data))

    if ranking == "growth":
        stock_data.sort(key=lambda x: x[1], reverse=True)
        winners = stock_data[:33]
        for i, winner in enumerate(winners):
            index, ret, std = winner
            price = current_hold_prices[index][-1]
            number_of_shares = math.floor((5000/(i + 1)) / price)
            if ret > 0:
                position[index] = number_of_shares
    elif ranking == "hybrid":
        array = np.array(stock_data)
        df = pd.DataFrame(array, columns=("index", "returns", "std", "model"))
        x = np.linspace(0, 1, 1000)
        y = 1 + 0.5 * x
        y_neg = -1 * y
        y_zero = np.zeros(1000)

        hold = []
        short = []
        dead_zone = []
        for i, (x_value, y_value) in enumerate(zip(df["std"], df["returns"])):
            expected_y_positive = 1 + 0.25 * x_value
            expected_y_negative = -1 * expected_y_positive
            if y_value >= expected_y_positive:
                hold.append((i, (y_value) / x_value))
            elif y_value <= expected_y_negative:
                short.append((i, (y_value) / x_value))
            else:
                dead_zone.append((i, y_value / x_value))

        positions = get_position_size(hold, short)

        for (index, money) in positions:
            position[index] = math.floor(money/current_hold_prices[index][-1])

        # fig, ax = plt.subplots()
        # ax.scatter(df["std"], df["returns"])
        # ax.scatter(x, y, marker='_')
        # for i, (x_1, y_1) in enumerate(zip(df["std"], df["returns"])):
        #     ax.annotate(i, (x_1, y_1))
        # ax.scatter(x, y_neg, marker='_')
        # ax.scatter(x, y_zero, marker='_')
        # plt.show()
    elif ranking == "regression":
        df = pd.DataFrame(np.array(stock_data), columns=("index", "returns", "std", "r2"))

        positions = []
        for i, (x_value, y_value) in enumerate(zip(df["r2"], df["returns"])):
            if x_value > cut_off_r:
                if y_value > 0:
                    positions.append((i, ratio_function(y_value) * (x_value - cut_off_r) * 10000 * (1/cut_off_r)))
                elif y_value < 0:
                    positions.append((i, ratio_function(y_value) * (x_value - cut_off_r) * 10000 * (1/cut_off_r)))
        # Plotting
        plot = False
        if plot:
            fig, ax = plt.subplots()
            ax.scatter(df["r2"], df["returns"])
            ax.scatter(np.linspace(0.5, 0.5, 1000), np.linspace(-15, 15, 1000), marker='|')
            for i, (x_1, y_1) in enumerate(zip(df["r2"], df["returns"])):
                ax.annotate(i, (x_1, y_1))
            ax.scatter(np.linspace(0, 1, 1000), np.zeros(1000), marker='_')
            plt.show()
            
        for (index, money) in positions:
            position[index] = math.floor(money/current_hold_prices[index][-1])

    # update_position(position, prcSoFar)

    return position