import math
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

data_history = 66
np.seterr("ignore")


class Stock:

    def __init__(self, stock_number, historic_price):
        self.price = historic_price[-1]
        self.number = stock_number
        self.historic_prices = historic_price
        self.training_days = []
        self.vol_training = []
        self.linear_model = LinearRegression()
        self.current_day = len(historic_price)
        self.scaled_stocks = []
        self.vibe = 0
        self.z = 0
        self.is_vol = False
        self.vol = 0

    def set_vibe_training_days(self, days):
        """
        Sets the vibe training days
        """
        self.training_days = self.historic_prices[-days:]

    def set_historic_days(self, hd):
        """
        Sets the amount of days we have access too
        """
        self.historic_prices = hd

    def is_volatile(self, days):
        """
        Checks if the stock is volatile
        :param days: Amount of days to check over
        :return: None
        """
        self.vol_training = self.historic_prices[-days:]
        total_negative_change = 0
        total_positive_change = 0
        for i in range(1, days):
            change = self.vol_training[i] - self.vol_training[i - 1]
            if change > 0:
                total_positive_change += change
            else:
                total_negative_change += change
        if (total_positive_change - total_negative_change) / self.price > 0.15:
            self.is_vol = True
        self.vol = (total_positive_change - total_negative_change) / self.price

    def update_model(self, days):
        """
        Update the linear regression model being used
        :param days: Linear regression model time period from today's day
        :return: None
        """
        self.linear_model = LinearRegression()
        x = np.array(list(range(self.current_day - days, self.current_day))).reshape(-1, 1)
        self.linear_model.fit(x, self.scaled_stocks[-days:])

    def get_r_squared(self, days):
        """
        Return the r2 value of the current stocks linear model
        :param days: The amount of time to check the r2 value on the LR
        :return: None
        """
        x = np.array(list(range(self.current_day - days, self.current_day))).reshape(-1, 1)
        r2 = self.linear_model.score(x, self.scaled_stocks[-days:])
        return r2

    def scale_stocks(self):
        """
        Scales all stocks down for easier computations
        """
        stock = []
        start_price = self.training_days[0]
        for price in self.training_days[1:]:
            stock.append(price / start_price)
        self.scaled_stocks = stock

    def update_vibe(self, periods, coefficients, r2_cutoffs):
        """
        Calculates the 'vibe' of the stock at the current point in time
        :param periods: Time periods
        :param coefficients: Weighting of each time period
        :param r2_cutoffs: r2 cutoff necessary for each period to even be considered for a position
        :return: none
        """
        total_vibe = 0
        for (period, coefficient, r2_cutoff) in zip(periods, coefficients, r2_cutoffs):
            self.update_model(period)
            if self.get_r_squared(period) < r2_cutoff:
                self.vibe = 0
                return
            else:
                total_vibe += 10 * coefficient * self.linear_model.coef_ * self.get_r_squared(period)
        self.vibe = (total_vibe / sum(periods)) * 1000

    def get_today_investment(self):
        """
        :return: returns the amount to invest in one stock based on the stocks variables
        """
        if self.is_vol:
            if self.z > 1:
                return -10000
            elif self.z < -1:
                return 10000
            else:
                return 0
        else:
            if self.vibe > 5:
                return 5000
            elif self.vibe > 4:
                return 7000
            elif self.vibe > 3:
                return 7500
            elif self.vibe > 2:
                return 9000
            elif self.vibe > 1:
                return 9500
            else:
                return 0

    def set_z_value(self):
        """
        Sets Z value of stock
        """
        self.z = (self.price - np.mean(self.vol_training)) / np.std(self.vol_training)


def write_csv(position):
    """
    Function creates a csv file if one doesn't exist and writes today's positions to be used for tomorrow
    :param position: Positions to store for tomorrow
    :return: None
    """
    data_frame = pd.DataFrame(position).T
    if os.path.isfile("./previous_days.csv"):
        df = pd.read_csv("./previous_days.csv")
        df.loc[len(df)] = position
        os.remove("./previous_days.csv")
        df.to_csv("./previous_days.csv", index=False)
    else:
        data_frame.to_csv("./previous_days.csv", index=False)


def getMyPosition(prcSoFar):
    """
    Function goes through each stock and evaluates if a position should be held or is currently being held. After which
    if it is determined that a position would be advantageous, the array at the stocks index is updated to reflect.
    :param prcSoFar: Prices of the index at a given day where day == len(prcSoFar)
    :return: len(prcSoFar) dimensional array where each [index] represents a position in 'index' stock
    """
    position = np.zeros(100)

    # Clean previous memory
    if prcSoFar.shape[1] == 1 and os.path.isfile("./previous_days.csv"):
        os.remove("./previous_days.csv")
        write_csv(position)
    elif prcSoFar.shape[1] == 1 and not os.path.isfile("./previous_days.csv"):
        write_csv(position)

    first_buy_day = (prcSoFar.shape[1] - data_history) - 1

    # If the amount of data current avialable does not meat our constant value return no position in any stock
    if prcSoFar.shape[1] <= data_history:
        return position
    yesterday_positions = pd.read_csv("./previous_days.csv")

    # Data Setting Up
    stocks = []
    for i in range(100):
        vibe_periods = [15, 30, 65]
        vibe_coefficients = [4, 1, 4]
        vibe_r2_cutoffs = [0.5, 0.5, 0.8]

        # Create a new stock object and initialize values
        new_stock = Stock(i, prcSoFar[i])
        new_stock.is_volatile(10)

        # For stocks that are NOT volatile we have set a 3 day holding period minimum
        if not new_stock.is_vol:
            if first_buy_day % 3 != 0:
                new_price = prcSoFar[:, :(prcSoFar.shape[1] - (first_buy_day % 3))]
                new_stock.set_historic_days(new_price[i])
                new_stock.is_volatile(10)

        new_stock.set_vibe_training_days(66)
        new_stock.scale_stocks()
        new_stock.set_z_value()
        new_stock.update_vibe(vibe_periods, vibe_coefficients, vibe_r2_cutoffs)
        stocks.append(new_stock)

    # Method
    for index in range(100):
        investment = stocks[index].get_today_investment()
        position[index] = math.floor(investment / prcSoFar[index][-1])

        # Checks if we currently have a position and if so will not update the position as to reduce
        # constantly updating position sizes (non volatile stock check only)
        if yesterday_positions.iloc[-1][f"{index}"] != 0 and not stocks[index].is_vol and position[index] > 0:
            if (yesterday_positions.iloc[-1][f"{index}"] * prcSoFar[index][-1]) > 10000 or \
                    (yesterday_positions.iloc[-1][f"{index}"] * prcSoFar[index][-1]) < -10000:
                position[index] = math.floor(9500 / prcSoFar[index][-1])
            else:
                position[index] = yesterday_positions.iloc[-1][f"{index}"]

    # Store today's position as for tomorrow
    write_csv(position)
    return position
