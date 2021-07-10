#!/usr/bin/envblah python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Oliver_Momentum_Strat import getMyPosition as getPosition
from plots_and_tools import get_the_market

nInst = 0
nt = 0
commRate = 0.0050
dlrPosLimit = 10000
holding_period = 1
data_history = 71
ranking = "regression"
cut_off_max = 10
cut_off_min = 5
cut_off_r = 0.5
std_days = 5

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return df.values.T


pricesFile = "./prices250.txt"
prcAll = loadPrices(pricesFile)
print(prcAll.shape)


def plot_price(prices):
    x = list(range(0, 250))
    index = prcAll[std_days]
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
def calcPL(prcHist, parameters, hyper=None):
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
        newPosOrig = getPosition(prcHistSoFar, parameters, hyper)
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
    return (plmu, ret, annSharpe, totDVolume)


def adjust_hyper_parameters(r1, r2, r3):
    result = []
    for o in r1:
        for p in r2:
            for q in r3:
                print(o, p, q)
                hyper = ([15, 30, 50], [4, 2, 1], [o, p, q])
                (meanpl, ret, sharpe, dvol) = calcPL(prcAll, parameters, hyper)
                print(meanpl, ret, sharpe, dvol)
                print("==================\n")
                result.append(((o, p, q), (meanpl, ret, sharpe, dvol)))

    result.sort(key=lambda x: x[1][2], reverse=True)
    return result


parameters = (holding_period, data_history, cut_off_max, cut_off_min, cut_off_r, std_days)


# rs1 = [0.4, 0.5, 0.6, 0.7, 0.8]
# rs2 = [0.4, 0.5, 0.6, 0.7, 0.8]
# rs3 = [0.5, 0.6, 0.7]
# hyper = adjust_hyper_parameters(rs1, rs2, rs3)
# print(hyper)

# print(params)

hyper_check = ([15, 30, 65], [4, 0, 4], [0.5, 0, 0.8])

(meanpl, ret, sharpe, dvol) = calcPL(prcAll, parameters, hyper_check)
print_results(meanpl, ret, sharpe, dvol)
plot_price(values_over_data)
