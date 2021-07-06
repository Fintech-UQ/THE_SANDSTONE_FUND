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
data_history = 24 * 5 + 1
ranking = "regression"
cut_off_max = 10
cut_off_min = 5
cut_off_r = 0.5
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


def adjust_hyper_parameters(holding_range, history_range, cut_max_range, cut_min_range, cut_r_range, std_days_range):
    result = []
    for i in range(holding_range[0], holding_range[1] + 1):
        for j in range(history_range[0], history_range[1] + 1):
            for k in cut_r_range:
                for m in range(cut_max_range[0], cut_max_range[1] + 1):
                    for n in range(cut_min_range[0], cut_min_range[1] + 1):
                        for o in range(std_days_range[0], std_days_range[1] + 1):
                            if m > n and o <= j:
                                print("Starting: ", i, j * 5, m, n, k, o)
                                parameters = (i, j * 5, m, n, k, o * 5)
                                (meanpl, ret, sharpe, dvol) = calcPL(prcAll, parameters)
                                result.append(((i, j, k, m, n), (meanpl, ret, sharpe, dvol)))
                                # print("Finished: ", i, j, k, m, n)
    result.sort(key=lambda x: x[1][1], reverse=True)
    return result


parameters = (holding_period, data_history, cut_off_max, cut_off_min, cut_off_r, std_days)

# params = adjust_hyper_parameters((1, 5), (1, 40), (8, 12), (2, 6), (0.4, 0.45, 0.5, 0.55, 0.6), (1, 40))

# print(params[:10])

(meanpl, ret, sharpe, dvol) = calcPL(prcAll, parameters)
print_results(meanpl, ret, sharpe, dvol)
