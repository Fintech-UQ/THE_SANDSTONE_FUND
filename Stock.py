from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import pandas as pd
import numpy as np


class Stock:

    def __init__(self, stock_number, historic_price):
        self.price = historic_price[-1]
        self.number = stock_number
        self.historic_prices = historic_price
        self.training_days = []
        self.linear_model = LinearRegression()
        self.current_day = len(historic_price)
        self.scaled_stocks = []
        self.vibe = 0

    def get_price(self):
        return self.price

    def update_price(self, price):
        self.price = price

    def get_historic_prices(self):
        return self.historic_prices

    def get_previous_days(self, days):
        return self.historic_prices[-days:]

    def set_training_days(self, days):
        self.training_days = self.historic_prices[-days:]

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

    def update_vibe(self, periods, coefficients, r2_cutoffs):
        total_vibe = 0
        for (period, coefficient, r2_cutoff) in zip(periods, coefficients, r2_cutoffs):
            self.update_model(period)
            if self.get_r_squared(period) < r2_cutoff:
                self.vibe = 0
                return
            total_vibe += coefficient * self.linear_model.coef_ * 10 * self.get_r_squared(period)
        self.vibe = (total_vibe / sum(periods)) * 1000

    def get_vibe(self):
        return self.vibe

    def get_today_investment(self):
        if self.vibe > 5:
            return 8000
        elif self.vibe > 4:
            return 7500
        elif self.vibe > 3:
            return 7000
        elif self.vibe > 2:
            return 6000
        elif self.vibe > 1:
            return 5000
        # elif self.vibe < -1:
        #     return -5000
        # elif self.vibe < -2:
        #     return -6000
        # elif self.vibe < -3:
        #     return -7000
        # elif self.vibe < -4:
        #     return -7500
        # elif self.vibe < -5:
        #     return -8000
        else:
            return 0

