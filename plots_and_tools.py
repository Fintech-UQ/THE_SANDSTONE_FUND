import pandas as pd
import matplotlib.pyplot as plt
import numpy as np









def get_the_market(prHst):

    # get the intial amount of shares owned by puting one dolar in each of them 
    # the total value should be 250
    # then for each day plot the value 
    number_of_shares_owned_for_each_stock = []
    for stock in prHst:
        #add the number of shares owned on day 1 (index 0) if $1 is invested
        number_of_shares_owned_for_each_stock.append(1/stock[0])

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
            percent_change_for_each_day.append(percent_change_today)
        average_change_this_week = np.average(percent_change_for_each_day)
        percent_change_avg_every_5_days.append(average_change_this_week)
    x = list(range(0, len(percent_change_avg_every_5_days)))
    plt.plot(x, percent_change_avg_every_5_days)
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


df=pd.read_csv('prices250.txt', sep='\s+', header=None, index_col=None)
# the T attribute swaps the rows and columns so the rows are now the stock prices
data = df.values.T

plot_average_market_percent_change(data)