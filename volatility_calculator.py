
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from plots_and_tools import get_the_market as get_market
from scipy.stats import pearsonr, spearmanr, kendalltau

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

def get_returns_of_stock(stock, number_of_days_to_take_returns_over):
     n = number_of_days_to_take_returns_over
     returns = []
     for i in range(0, len(stock) - n, n):
        # percent change for each day this week
        percent_return = ((stock[i+5] - stock[i])/stock[i+5])*100
        returns.append(percent_return)
     return returns
 
# this indicates how volatile the stock is relative to the market
# is calculated by getting 
# ((the std of the stocks returns)/(the std of the markets returns)) * pearson_correlation_coeficient
def stocks_beta(prHst):
    # standard deviation of returns for each week
    # 
    # comparison_period = len(prHst[0])
    number_of_days_to_take_returns_over = 20
    betas = []
    market_returns = get_returns_of_stock(get_market(prHst), number_of_days_to_take_returns_over)
    market_std = np.std(market_returns)
    for i in range(len(prHst)):
        stock_returns = get_returns_of_stock(prHst[i], number_of_days_to_take_returns_over)
        stock_std = np.std(stock_returns)
        cor, p = kendalltau(stock_returns, market_returns)
        # indicating how volatile it is relative to the market
        beta = (stock_std/market_std) * cor
        betas.append((i, beta))
    ranking_values = []
    for vals in betas:
        ranking_values.append((vals[0], abs(vals[1]-1)))
    ranking_values.sort(key=lambda x: x[1], reverse=True)
    #return ranking_values
    betas.sort(key=lambda x: x[1], reverse=True)
    print(betas)
    return ranking_values

def plot_stock_volatilities(prHst, stock_volatilities):
    scaled_stocks = get_scale_stocks(prHst)
    for i in range(3):
        stock_index = stock_volatilities[i][0]
        prices = scaled_stocks[stock_index]
        avgpric = get_avg_every_5_days(prices)
        x = list(range(0, len(avgpric)))
        plt.plot(x, avgpric)

    #plot the market
    market = get_scaled_market(prHst)
    plt.plot(get_avg_every_5_days(market), 'red')
    plt.title("Top 3 least volatile")
    plt.xlabel("days")
    plt.ylabel("average daily percent change")
    plt.show()
    
def get_avg_every_5_days(arr):
    every_5_days = []
    for i in range(0,len(arr) - 5, 5):
        every_5_days.append(np.average(arr[i:i+5]))
    return every_5_days


def volatility_return_ranker(prHist):
    num_days = 30
    vol = []
    for i in range(len(prHist)):
        stock = prHist[i]
        past_n_days = stock[len(prHist) - num_days:]
        
        vol.append((i, np.std(past_n_days)*math.sqrt(num_days)))
    vol.sort()
    print(vol)
    # ((percent return)/(vol*som_constant) - 1)*something
    # dead_zone is volatility*gradient = return 


if __name__ == "__main__":
    #percent_change_volatility(data)
    #stocks_betas = stocks_beta(data)
    #plot_stock_volatilities(data, stocks_betas)
    volatility_return_ranker(data)