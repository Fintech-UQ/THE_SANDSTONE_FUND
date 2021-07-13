import math
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from Stock import *
import matplotlib.pyplot as plt
from plots_and_tools import get_the_market
import numpy as np
data_history = 10

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

    # Training Period
    if prcSoFar.shape[1] <= data_history:
        return position
    else:
        if prcSoFar.shape[1] == data_history + 1:
            write_csv(position)
        yesterday_positions = pd.read_csv("./previous_days.csv")

    # Holding Period
    # if previous_hold_day % 2 != 0:
    #     prcSoFar = prcSoFar[:, :(prcSoFar.shape[1] - (previous_hold_day % 2))]

    # Data Setting Up
    stocks = []
    for i in range(100):
        new_stock = VolStock(i, prcSoFar[i])
        new_stock.set_training_days(10)
        new_stock.set_running_avg()
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

nInst = 0
nt = 0
commRate = 0.0050
dlrPosLimit = 10000

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return df.values.T


pricesFile = "./prices250.txt"
prcAll = loadPrices(pricesFile)

def plot_price(prices):
    x = list(range(0, 250))
    index = prcAll[84]
    # index = get_the_market(prcAll)
    fig, ax = plt.subplots()
    ax.plot(x, prices, color="red", marker="o", markersize=2.5)
    ax.set_ylabel("Earnings $", fontsize=14)
    ax.set_xlabel("Trading Days", fontsize=14)
    ax2 = ax.twinx()
    ax2.plot(x, index, color="blue", marker="o", markersize=2.5)
    ax2.set_ylabel("Index Price $", fontsize=14)
    plt.show()

def print_results(mean_pl, returns, sharpe_value, d_vol):
    print("=====")
    print("mean(PL): %.0lf" % mean_pl)
    print("return: %.5lf" % returns)
    print("annSharpe(PL): %.2lf " % sharpe_value)
    print("totDvolume: %.0lf " % d_vol)


values_over_data = []


def calcPL(prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolume0 = 0
    totDVolume1 = 0
    frac0 = 0.
    frac1 = 0.
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    for t in range(1, 251):
        prcHistSoFar = prcHist[:, :t]
        newPosOrig = getMyPosition(prcHistSoFar)
        curPrices = prcHistSoFar[:, -1]
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.array([int(p) for p in np.clip(newPosOrig, -posLimits, posLimits)])
        deltaPos = newPos - curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume0 = np.sum(dvolumes[:50])
        dvolume1 = np.sum(dvolumes[50:])
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        totDVolume0 += dvolume0
        totDVolume1 += dvolume1
        comm = dvolume * commRate
        cash -= curPrices.dot(deltaPos) + comm
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        todayPLL.append(todayPL)
        value = cash + posValue
        values_over_data.append(value)
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
            frac0 = totDVolume0 / totDVolume
            frac1 = totDVolume1 / totDVolume
        print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf frac0: %.4lf frac1: %.4lf" % (
        t, value, todayPL, totDVolume, ret, frac0, frac1))
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = 16 * plmu / plstd
    return (plmu, ret, annSharpe, totDVolume, value)

(meanpl, ret, sharpe, dvol, value) = calcPL(prcAll)
print_results(meanpl, ret, sharpe, dvol)
plot_price(values_over_data)
