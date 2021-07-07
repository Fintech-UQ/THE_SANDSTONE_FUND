
import numpy as np
import math


def get_prices_until_the_last_monday(prcSoFar):
    '''
    determine the number of days since monday. this is because our position
    only changes every monday so we need to calculate what our position
    is if it where to be calculated from the last moday
    '''
    tmp = len(prcSoFar[0])%5 - 1
    number_of_days_since_moday = 4 if tmp == -1 else tmp

    # gets the index of the most recent monday
    current_index = len(prcSoFar[0]) - 1
    monday_index = current_index - number_of_days_since_moday

    # gets an array that includes the prices up until the most recent moday
    prices_up_until_monday = prcSoFar.T[:monday_index].T
    return prices_up_until_monday


def get_the_market(prHst):
    # get the intial amount of shares owned by puting one dolar in each of them 
    # the total value should be 250
    # then for each day plot the value 
    number_of_shares_owned_for_each_stock = []
    for stock in prHst:
        #add the number of shares owned on day 1 (index 0) if $1 is invested
        number_of_shares = 1/stock[0]
        number_of_shares_owned_for_each_stock.append(number_of_shares)
    # this is value not number of stocks

    return number_of_shares_owned_for_each_stock


def getMyPosition (prcSoFar):
    return get_the_market(prcSoFar)
    # if 13 weeks have not passed then we do not take a position
    if len(prcSoFar) <= 65:
        return np.zeros(100)
    # up until and including the last monday
    prices_up_until_monday = get_prices_until_the_last_monday(prcSoFar)

    # rank all of the stocks make a list of tuples (index, return)
    '''
    our momentum strategy uses past 12-week return minus 
    most recent month’s return to rank stocks
    '''
    # gets a list of all the stocks and their returns 
    stock_returns = []
    for i in range(len(prices_up_until_monday)):
        stock = prices_up_until_monday[i]
        index = len(stock) - 1
        # the monday 13 weeks ago
        start_monday_index = index - 65
        # the friday 12 weeks after 
        end_friday_index = index - 6

        # percent return 
        percent_return = ((stock[end_friday_index] - stock[start_monday_index])/stock[end_friday_index])*100
        period = stock[start_monday_index:end_friday_index]
        vol = np.std(period)*len(period)
       
        momentum_return = percent_return
        stock_returns.append((i, momentum_return, vol))
    
    # sorts them in order of their return 
    stock_returns.sort(key = lambda x: x[1], reverse=True) 
    print('THE VERY BEGINNING')
    print('THE VERY BEGINNING')
    print('THE VERY BEGINNING')
    print('THE VERY BEGINNING')
    print(stock_returns)

    '''
    make a list of zeros then loop through the top 33 ranked stocks
    and set each value to be the number of shares which will be 10,000/price
    '''
    
    position = np.zeros(100)

    for i in range(10):
        stock_index = stock_returns[i][0]
        percent_return = stock_returns[i][1]
        price = prices_up_until_monday[stock_index][-1]
        number_of_shares = math.floor(5000/price)
        percent_return = stock_returns[i][1]
        if percent_return > 0:
            position[stock_index] = number_of_shares
    
    
    return position








