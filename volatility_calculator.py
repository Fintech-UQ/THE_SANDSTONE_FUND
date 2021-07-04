
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from plots_and_tools import get_the_market as get_market

# percent change 
# correlation coeficent for linear and quadratic 
# comparison to the 
# volatility is calculated from n number of past weeks
# beta is a calculation of volatility with respect to a benchmark 
# use variance as a measure of how far the price of the asset may deviate from the mean


df=pd.read_csv('prices250.txt', sep='\s+', header=None, index_col=None)
# the T attribute swaps the rows and columns so the rows are now the stock prices
data = df.values.T

# percent change volatility calculator 
def percent_change_volatility(prHst):
    scaled_stocks = get_scale_stocks(prHst)
    # the average daily percent changes for all stocks 
    average_daily_percent_changes = []
    for j in range(len(scaled_stocks)):
        stock = scaled_stocks[j]
        days = len(stock)
        daily_percent_changes = []
        for i in range(1, days):
            percent_change_today = ((stock[i] - stock[i-1])/stock[i])*100
            daily_percent_changes.append(abs(percent_change_today))
        average_daily_percent_changes.append((j, np.average(daily_percent_changes)))

    average_daily_percent_changes.sort(key=lambda x: x[1])
    return average_daily_percent_changes
    # for i in range(3):
    #     stock_index = average_daily_percent_changes[i][0]
    #     prices = scaled_stocks[stock_index]
    #     x = list(range(0, len(prices)))
    #     plt.plot(x, prices)
    # most_volatile = scaled_stocks[average_daily_percent_changes[-1][0]]
    # plot the most volatile
    # x = list(range(0, len(most_volatile)))
    # plt.plot(x, most_volatile,'red')

    # plot the market
    # market = get_scaled_market(prHst)
    # plt.plot(x, market,'red')
    # plt.title("Top 3 least volatile")
    # plt.xlabel("days")
    # plt.ylabel("average daily percent change")
    # plt.show()


def get_scaled_market(prHst):
    market = get_market(prHst)
    scale_factor = market[0]
    scaled_market = []
    for price in market:
        scaled_market.append(price/scale_factor)
    return scaled_market

def get_scale_stocks(prHst):
    scaled_stocks = []
    for stock in prHst:
        scale_factor = stock[0]
        scaled_stock = []
        for price in stock:
            scaled_stock.append(price/scale_factor)
        scaled_stocks.append(scaled_stock)
    return scaled_stocks
    
if __name__ == "__main__":
    percent_change_volatility(data)