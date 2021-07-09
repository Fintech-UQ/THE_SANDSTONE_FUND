import pandas as pd
import matplotlib.pyplot as plt

def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    nt, nInst = df.values.shape
    return df.values.T


pricesFile = "./prices250.txt"
prcAll = loadPrices(pricesFile)


def plot_price(prices, stock,  consider=False):
    if consider:
        x = list(range(0, 250))
        index = prcAll[18]
        fig, ax = plt.subplots()
        ax.plot(x, prices, color="red", marker="o", markersize=2.5)
        ax.set_ylabel("Earnings $", fontsize=14)
        ax.set_xlabel("Trading Days", fontsize=14)
        ax2 = ax.twinx()
        ax2.plot(x, index, color="blue", marker="o", markersize=2.5)
        ax2.set_ylabel("Index Price $", fontsize=14)
        plt.show()
    else:
        x = list(range(0, 250))
        index = prcAll[stock]
        plt.plot(x, index, color="blue", marker="o", markersize=2.5)
        plt.savefig(f"./Stocks/Stock {stock}.png")
        plt.cla()

for stock in range(100):
    plot_price([1], stock)