
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Oliver_Momentum_Strat import ratio_function
from plots_and_tools import get_the_market as get_market
import random

df=pd.read_csv('prices250.txt', sep='\s+', header=None, index_col=None)
# the T attribute swaps the rows and columns so the rows are now the stock prices
data = df.values.T


# ratio function test
val1 = ratio_function(-8)
if round(val1,1) != -0.6:
    raise Exception("ratio function no worky")

# getMyPositionTest 

def getMyPosition(data):
    portfolio = np.zeros(100)
    portfolio[42] = -1 - random.randint(0,1)
    return portfolio
