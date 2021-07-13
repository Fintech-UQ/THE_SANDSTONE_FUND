import math
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
data_history = 66


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
        self.std = 0
        self.is_vol = False

    def set_training_days(self, days):
        self.training_days = self.historic_prices[-days:]

    def is_volatile(self, days):
        self.vol_training = self.historic_prices[-days:]
        negative = 0
        positive = 0
        for i in range(1, days):
            change = self.vol_training[i] - self.vol_training[i - 1]
            if change > 0:
                positive += change
            else:
                negative += change
        if (positive - negative) / self.price > 0.1:
            self.is_vol = True


    def update_model(self, days):
        self.linear_model = LinearRegression()
        x = np.array(list(range(self.current_day - days, self.current_day))).reshape(-1, 1)
        self.linear_model.fit(x, self.scaled_stocks[-days:])

    def get_r_squared(self, days):
        x = np.array(list(range(self.current_day - days, self.current_day))).reshape(-1, 1)
        r2 = self.linear_model.score(x, self.scaled_stocks[-days:])
        return r2

    def scale_stocks(self):
        stock = []
        start_price = self.training_days[0]
        for price in self.training_days[1:]:
            stock.append(price / start_price)
        self.scaled_stocks = stock

    def check_vibe(self, periods, coefficients, r2_cutoffs):
        total_vibe = 0
        for (period, coefficient, r2_cutoff) in zip(periods, coefficients, r2_cutoffs):
            total_vibe += 10 * coefficient * self.linear_model.coef_ * self.get_r_squared(period)
        return (total_vibe / sum(periods)) * 1000

    def update_vibe(self, periods, coefficients, r2_cutoffs):
        total_vibe = 0
        for (period, coefficient, r2_cutoff) in zip(periods, coefficients, r2_cutoffs):
            self.update_model(period)
            if self.get_r_squared(period) < r2_cutoff:
                self.vibe = 0
                return
            else:
                total_vibe += 10 * coefficient * self.linear_model.coef_ * self.get_r_squared(period)
        self.vibe = (total_vibe / sum(periods)) * 1000

    def get_long_position_size(self):
        if self.vibe > 5:
            return 5000
        elif self.vibe > 4:
            return 7000
        elif self.vibe > 3:
            return 7500
        elif self.vibe > 2:
            return 9000
        else:
            return 9500

    def get_today_investment(self):
        if self.is_vol:
            if self.std > 1:
                return -9500
            elif self.std < -1:
                return 9500
            else:
                return 0
        else:
            if self.vibe > 1:
                return self.get_long_position_size()
            else:
                return 0

    def set_std(self):
        self.std = (self.price - np.mean(self.vol_training)) / np.std(self.vol_training)


def write_csv(position):
    data_frame = pd.DataFrame(position).T
    if os.path.isfile("./previous_days.csv"):
        df = pd.read_csv("./previous_days.csv")
        df.loc[len(df)] = position
        os.remove("./previous_days.csv")
        df.to_csv("./previous_days.csv", index=False)
    else:
        data_frame.to_csv("./previous_days.csv", index=False)


def getMyPosition(prcSoFar):
    position = np.zeros(100)

    # CSV of previous day positions used as memory for commission-cost management
    if prcSoFar.shape[1] == 1 and os.path.isfile("./previous_days.csv"):
        os.remove("./previous_days.csv")

    # Yesterday's hold index
    previous_hold_day = (prcSoFar.shape[1] - data_history) - 1
    if prcSoFar.shape[1] == 1:
        write_csv(position)
    yesterday_positions = pd.read_csv("./previous_days.csv")

    # Holding Period
    if previous_hold_day % 3 != 0:
        prcSoFar = prcSoFar[:, :(prcSoFar.shape[1] - (previous_hold_day % 3))]

    # Data Setting Up
    stocks = []
    for i in range(100):
        vibe_periods = [15, 30, 65]
        vibe_coefficients = [4, 1, 4]
        r2_cutoffs = [0.5, 0.5, 0.8]
        new_stock = Stock(i, prcSoFar[i])
        if prcSoFar.shape[1] > 10:
            new_stock.is_volatile(10)
            if new_stock.is_vol and prcSoFar.shape[1] < data_history:
                new_stock.set_training_days(10)
                new_stock.scale_stocks()
                new_stock.set_std()
            elif prcSoFar.shape[1] >= data_history:
                new_stock.set_training_days(66)
                new_stock.scale_stocks()
                new_stock.set_std()
                new_stock.update_vibe(vibe_periods, vibe_coefficients, r2_cutoffs)
        stocks.append(new_stock)

    # Method
    for index in range(100):
        # if index < 49:
        investment = stocks[index].get_today_investment()
        position[index] = math.floor(investment / prcSoFar[index][-1])
        if yesterday_positions.iloc[-1][f"{index}"] != 0 and position[index] > 0:
            if (yesterday_positions.iloc[-1][f"{index}"] * prcSoFar[index][-1]) > 10000:
                position[index] = math.floor(9500 / prcSoFar[index][-1])
            else:
                position[index] = yesterday_positions.iloc[-1][f"{index}"]

    write_csv(position)
    return position
