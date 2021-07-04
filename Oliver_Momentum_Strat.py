import numpy as np
import math
from sklearn.linear_model import LinearRegression

momentumSize = 24 * 5


def get_prices_until_the_last_monday(prcSoFar):
    tmp = prcSoFar.shape[1] % 5 - 1
    number_of_days_since_moday = 4 if tmp == -1 else tmp

    # gets the index of the most recent monday
    current_index = prcSoFar.shape[1] - 1
    monday_index = current_index - number_of_days_since_moday

    # gets an array that includes the prices up until the most recent moday
    prices_up_until_monday = prcSoFar.T[:monday_index].T
    return prices_up_until_monday


def get_market(prHst):
    number_of_shares_owned_for_each_stock = []
    for stock in prHst:
        # add the number of shares owned on day 1 (index 0) if $1 is invested
        number_of_shares_owned_for_each_stock.append(1 / stock[0])

    # an array for the market value for each day of $1 in each stonk
    the_market_value = []
    # looping through each day
    for day in prHst.T:
        value_for_the_day = 0
        # loops through the prices of each stock for that day
        for i in range(len(day)):
            value_for_the_day += day[i] * number_of_shares_owned_for_each_stock[i]
        the_market_value.append(value_for_the_day)
    return the_market_value


def get_rankings(prices_up_until_monday):
    stocks = []
    for i, stock in enumerate(prices_up_until_monday):
        index = len(stock) - 1
        start_monday_index = index - momentumSize
        end_friday_index = index - 6

        percent_return = ((stock[end_friday_index] - stock[start_monday_index]) / stock[end_friday_index]) * 100

        stocks.append((i, percent_return))

    stocks.sort(key=lambda x: x[1], reverse=True)
    return stocks


def get_weighted_rankings(winners, prcSoFar, mondayStuff):
    weeks = 10
    days = -10

    rankings = []
    for winner in winners:
        index, change = winner
        changes = []
        for i in range(weeks):
            percent_change = (mondayStuff[index][days + 4] - mondayStuff[index][days]) / mondayStuff[index][days + 4]
            changes.append(percent_change)
        avg = np.average(changes)
        rankings.append((index, avg))

    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings

    # Regression
    # train_x = np.array(range(10)).reshape(-1, 1)
    # test_x = np.array(range(11, 21)).reshape(-1, 1)
    # rankings = []
    # for winner in winners:
    #     index, change = winner
    #
    #     train_y = np.array(prcSoFar[index][-20:-10]).reshape(-1, 1)
    #     model = LinearRegression()
    #     model.fit(train_x, train_y)
    #     predicted = model.predict(test_x)
    #     real_y = prcSoFar[index][-10:]
    #
    #     sum_var = 0
    #     for i in range(len(predicted)):
    #         sum_var += abs(real_y[i] - predicted[i])
    #     var = sum_var / len(predicted)
    #
    #     rankings.append((index, var))
    # rankings.sort(key=lambda x: x[1], reverse=True)
    # return rankings


def update_position(currentPositions, prcSoFar):
    for i, position in enumerate(currentPositions):
        if position * prcSoFar[i][-1] > 10000:
            currentPositions[i] = math.floor(5000 / prcSoFar[i][-1])
            print(f"Sold {i} on day {prcSoFar.shape[1]}")


def getMyPosition(prcSoFar, i):
    '''
    :param prcSoFar: Price's at the current time
    :return: Vector of our current positions
    '''
    momentumSize = i * 5
    # Check if momentumSize weeks have passed
    if prcSoFar.shape[1] <= momentumSize:
        return np.zeros(100)

    prices_up_until_monday = get_prices_until_the_last_monday(prcSoFar)

    stock_returns = get_rankings(prices_up_until_monday)

    position = np.zeros(100)

    winners = stock_returns[:33]

    losers = stock_returns[67:]

    winners_evaluated = get_weighted_rankings(winners, prcSoFar, prices_up_until_monday)

    for i, winner in enumerate(winners):
        index, change = winner
        price = prices_up_until_monday[index][-1]
        number_of_shares = math.floor((5000/(i + 1)) / price)
        if change > 0:
            position[index] = number_of_shares

    # for i, loser in enumerate(losers):
    #     rank = 33 - i
    #     index, change = loser
    #     price = prices_up_until_monday[index][-1]
    #     number_of_shares = -1 * (math.floor((1000 / rank) / price))
    #     position[index] = number_of_shares

    update_position(position, prcSoFar)

    return position








