#!/usr/bin/envblah python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Oliver_Momentum_Strat import getMyPosition as getPosition

nInst = 0
nt = 0
commRate = 0.0050
dlrPosLimit = 10000
holding_period = 5
data_history = 24 * 5
ranking = "hybrid"
waiting_period = 6
no_stocks = 15
std_days = 60

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return df.values.T


pricesFile = "./prices250.txt"
prcAll = loadPrices(pricesFile)


def print_results(mean_pl, returns, sharpe_value, d_vol):
    print("=====")
    print("mean(PL): %.0lf" % mean_pl)
    print("return: %.5lf" % returns)
    print("annSharpe(PL): %.2lf " % sharpe_value)
    print("totDvolume: %.0lf " % d_vol)


def calcPL(prcHist, parameters):
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
        newPosOrig = getPosition(prcHistSoFar, parameters)
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


def adjust_hyper_parameters(holding_range, history_range, ranking_range, waiting_range, stock_range):
    result = []
    for i in range(holding_range[0], holding_range[1] + 1):
        for j in range(history_range[0], history_range[1] + 1):
            for k in ranking_range:
                for m in range(waiting_range[0], waiting_range[1] + 1):
                    for n in range(stock_range[0], stock_range[1]):
                        if m <= i:
                            # print("Starting: ", i, j, k, m, n)
                            parameters = (i, j * 5, k, m, n)
                            (meanpl, ret, sharpe, dvol) = calcPL(prcAll, parameters)
                            result.append(((i, j, k, m, n), (meanpl, ret, sharpe, dvol)))
                            # print("Finished: ", i, j, k, m, n)
    result.sort(key=lambda x: x[1][1], reverse=True)
    return result


parameters = (holding_period, data_history, ranking, waiting_period, no_stocks, std_days)

# params = adjust_hyper_parameters((1, 5), (1, 40), ("growth", "vol", "g.v"), (1, 10), (10, 40))
# print(params[:10])

(meanpl, ret, sharpe, dvol) = calcPL(prcAll, parameters)
print_results(meanpl, ret, sharpe, dvol)
