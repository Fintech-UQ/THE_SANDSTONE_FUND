import numpy as np
import math
from sklearn.linear_model import LinearRegression

holding_period = 1
data_history = 5
ranking = "growth"
waiting_period = 1
np.set_printoptions(suppress=True)
no_stocks = 15

def get_current_hold_prices(prcSoFar):
    tmp = prcSoFar.shape[1] % holding_period - 1

    days_since_hold = holding_period - 1 if tmp == -1 else tmp

    current_index = prcSoFar.shape[1] - 1
    monday_index = current_index - days_since_hold

    return prcSoFar.T[:monday_index].T

def get_rankings(prices_up_until_monday):
    stocks = []
    for i, stock in enumerate(prices_up_until_monday):
        index = len(stock) - 1
        start_monday_index = index - data_history
        end_friday_index = index - waiting_period
        percent_return = ((stock[end_friday_index] - stock[start_monday_index]) / stock[end_friday_index]) * 100

        stocks.append((i, percent_return))

    stocks.sort(key=lambda x: x[1], reverse=True)
    return stocks


def update_position(currentPositions, prcSoFar):
    for i, position in enumerate(currentPositions):
        if position * prcSoFar[i][-1] > 10000:
            currentPositions[i] = math.floor(5000 / prcSoFar[i][-1])
            print(f"Sold {i} on day {prcSoFar.shape[1]}")


def get_value(prcSoFar, positions):
    values = np.array(positions)
    for i, pos in enumerate(positions):
        values[i] = prcSoFar[i][-1] * values[i]
    return values


def set_parameters(parameters):
    global holding_period, waiting_period, data_history, ranking, no_stocks
    (holding_period, data_history, ranking, waiting_period, no_stocks) = parameters


def getMyPosition(prcSoFar, parameters):
    global holding_period, waiting_period, data_history, ranking, no_stocks
    set_parameters(parameters)
    position = np.zeros(100)

    if prcSoFar.shape[1] <= data_history:
        return np.zeros(100)

    current_hold_prices = get_current_hold_prices(prcSoFar)

    stock_returns = get_rankings(current_hold_prices)

    winners = stock_returns[:no_stocks]

    for i, winner in enumerate(winners):
        index, change = winner
        price = current_hold_prices[index][-1]
        number_of_shares = math.floor((8000/(i + 1)) / price)
        if change > 0:
            position[index] = number_of_shares
    update_position(position, prcSoFar)

    return position








