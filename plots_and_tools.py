import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression

def scale(prHst):
    stocks = []
    for old_stock in prHst:
        stock = []
        start_price = old_stock[0]
        for price in old_stock:
            stock.append(price/start_price)
        stocks.append(stock)
    return stocks



def get_the_market(prHst):
    # get the intial amount of shares owned by puting one dolar in each of them 
    # the total value should be 250
    # then for each day plot the value 
    number_of_shares_owned_for_each_stock = []
    for stock in prHst:
        #add the number of shares owned on day 1 (index 0) if $1 is invested
        number_of_shares_owned_for_each_stock.append(1)

    # an array for the market value for each day of $1 in each stonk
    the_market_value = []
    #looping through each day
    for day in prHst.T:
        value_for_the_day = 0
        # loops through the prices of each stock for that day
        for i in range(len(day)):
            value_for_the_day += day[i]*number_of_shares_owned_for_each_stock[i]
        the_market_value.append(value_for_the_day)
    
    return the_market_value


def plot_market(prHst):
    the_market_value = get_the_market(prHst)
    x = list(range(0, len(the_market_value)))
    plt.plot(x, the_market_value)
    plt.title("$1 in each stock from day 1 - 250")
    plt.xlabel("days")
    plt.ylabel("$1 portfolio value")
    print("start value: ", the_market_value[0], "end value: ", the_market_value[-1])
    print("value change: ", the_market_value[-1] - the_market_value[0], "percent_change: ",
     ((the_market_value[-1]- the_market_value[0])/the_market_value[-1])*100)
    plt.show()

# plots the average percent change of the market every 5 days
def plot_average_market_percent_change(prHst):
    the_market_values = get_the_market(prHst)
    percent_change_avg_every_5_days = []
    for i in range(1, len(the_market_values) -5, 5):
        # percent change for each day this week
        percent_change_for_each_day = []
        for j in range(i, i + 5):
            percent_change_today = ((the_market_values[j] - the_market_values[j-1])/the_market_values[j])*100
            percent_change_for_each_day.append(abs(percent_change_today))
        average_change_this_week = np.average(percent_change_for_each_day)
        percent_change_avg_every_5_days.append(average_change_this_week)
    x = list(range(0, len(percent_change_avg_every_5_days)))
    plt.bar(x, percent_change_avg_every_5_days)
    plt.title("Average Percent Change Every week (5 days)")
    plt.xlabel("weeks")
    plt.ylabel("Average percent change")
    plt.show()



def convert_txt_file_to_csv(txt_file_name, new_csv_file_name):
    # e.g. file_name = 'prices250.txt'
    # e.g. new_csv_file_name = 'prices250.csv'
    # NOTE: the txt file must be in the same directory as this file
    read_file=pd.read_csv(txt_file_name,sep='\s+', header=None, index_col=None)
    read_file.to_csv(new_csv_file_name, header=None,index=None)

def plot_stock(prHst, stock_index):
    y = prHst[stock_index]
    x = list(range(0, len(y)))
    plt.plot(x, y)
    plt.show()

def plot_scaled_stock(prHst, stock_index):
    scaled_stocks = scale(prHst)
    y = scaled_stocks[stock_index]
    x = list(range(0, len(y)))
    plt.plot(x, y)
    plt.show()


def get_line_of_best_fit_error(stock):
    xs = list(range(0, len(stock)))
    weights = np.polyfit(xs, stock, 1)
    model = np.poly1d(weights)
    y_predict = model(xs)
    for x in xs:
        error = stock[x] - y_predict[x]
    #return y_predict

def plot_vol_and_stock(prHst, stock_index):
    scaled_stocks = scale(prHst)
    y = scaled_stocks[stock_index]
    x = list(range(0, len(y)))
    plt.plot(x, y)
    #x_values = np.array(list(range(prcSoFar.shape[1] - std_days, prcSoFar.shape[1]))).reshape(-1, 1)
    regression_period = 120
    reg_vals_at_each_time_period = []
    time_periods = []
    for i in range(regression_period, len(y)):
        regression_values = y[i-regression_period:i]
        x_vals = list(range(i-regression_period, i))
        # reg_model = LinearRegression()
        # reg_model.fit(x_vals, regression_values)
        # r_squared = reg_model.score(x_vals, regression_values)
        correlation_matrix = np.corrcoef(x_vals, regression_values)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        reg_vals_at_each_time_period.append(r_squared)
        time_periods.append(i)

    plt.plot(time_periods, reg_vals_at_each_time_period)
    plt.show()
    #reg_model.score(x_values, prcSoFar[i][-std_days:])
def plotting_garbage():
        # Get the angles from 0 to 2 pie (360 degree) in narray object
    X = np.arange(0, math.pi*2, 0.05)
    
    # Using built-in trigonometric function we can directly plot
    # the given cosine wave for the given angles
    Y1 = np.sin(X)
    Y2 = np.cos(X)
    Y3 = np.tan(X)
    Y4 = np.tanh(X)
    
    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(1, 2)
    
    # For Sine Function
    axis[0].plot(X, Y1)
    axis[0].set_title("Sine Function")
    
    # For Cosine Function
    axis[1].plot(X, Y2)
    axis[1].set_title("Cosine Function")
    
    # # For Tangent Function
    # axis[1, 0].plot(X, Y3)
    # axis[1, 0].set_title("Tangent Function")
    
    # # For Tanh Function
    # axis[1, 1].plot(X, Y4)
    # axis[1, 1].set_title("Tanh Function")
    
    # Combine all the operations and display
    plt.show()

def get_vibe(prHst, stock_index):
    max_period = 120
    vibe_periods = [30, 60, 120]
    vibe_coeficients = [1, 2, 4]
    total_vibe = 0
    y = prHst[stock_index]
    n = len(y) - 1
    for period, coef in zip(vibe_periods, vibe_coeficients):
        y_period = y[n-period:n]
        x_period = list(range(n-period, n))
        slope, intercept = np.polyfit(x_period, y_period, 1)
        correlation_matrix = np.corrcoef(x_period, y_period)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        total_vibe += coef*slope*r_squared
    return total_vibe/sum(vibe_coeficients)
    



def plot_vibe(prHst, stock_index):
    figure, axis = plt.subplots(2, 1)
    scaled_stocks = scale(prHst)
    y = scaled_stocks[stock_index]
    # x = list(range(119, len(y)))
    x = list(range(0, len(y)))
    # from_120 = y[x[0]:x[-1]+1]
    axis[0].plot(x, y)
    max_period = 120
    vibe_periods = [30, 60, 120]
    vibe_coeficients = [4, 2, 1]
    vibes = []
    for i in range(max_period+1, len(y)):
        total_vibe = 0
        for j in range(len(vibe_periods)):
            y_period = y[i-vibe_periods[j]:i]
            x_period = list(range(i-vibe_periods[j], i))
            slope, intercept = np.polyfit(x_period, y_period, 1)
            correlation_matrix = np.corrcoef(x_period, y_period)
            correlation_xy = correlation_matrix[0,1]
            r_squared = correlation_xy**2
            total_vibe += vibe_coeficients[j]*slope*r_squared
        print(total_vibe)
        vibes.append(((total_vibe/sum(vibe_coeficients))*1000))
    
    x_vals = list(range(len(y) - len(vibes), len(y)))
    before_y = np.ones(250)
    before_x = list(range(0, len(before_y)))
    axis[1].plot(before_x, before_y)
    axis[1].plot(x_vals, vibes)
    plt.show()




    # regression_period = 120
    # max_period = 120
    # reg_vals_at_each_time_period = []
    # time_periods = []
    # for i in range(regression_period, len(y)):
    #     regression_values = y[i-regression_period:i]
    #     x_vals = list(range(i-regression_period, i))
    #     # reg_model = LinearRegression()
    #     # reg_model.fit(x_vals, regression_values)
    #     # r_squared = reg_model.score(x_vals, regression_values)
    #     correlation_matrix = np.corrcoef(x_vals, regression_values)
    #     correlation_xy = correlation_matrix[0,1]
    #     r_squared = correlation_xy**2
    #     reg_vals_at_each_time_period.append(r_squared)
    #     time_periods.append(i)


    # plt.plot(time_periods, reg_vals_at_each_time_period)
    # plt.show()


def get_line_of_best_fit(data, stock_index):
    stock = data[stock_index]
    y = stock[len(stock)-60:]
    x = list(range(len(y)-60, len(y)))
    slope, intercept = np.polyfit(x, y, 1)
    print(slope)


if __name__ == "__main__":
    df=pd.read_csv('prices250.txt', sep='\s+', header=None, index_col=None)
    # the T attribute swaps the rows and columns so the rows are now the stock prices
    data = df.values.T
    #plotting_garbage()
    plot_vibe(data, 41)
    for i in range(100):
        plot_vibe(data, i)

    #plot_vol_and_stock(data, 35)
    #get_line_of_best_fit(data, 5)
    #get_the_market(data)
    #plot_market(data)
    # greatest ret/vol then give it 5k
    # then multiply that by the 5k
    # if return is less then its volatility*gradient yeet it
    #plot_average_market_percent_change(data)