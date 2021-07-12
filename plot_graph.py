import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



r2_values = pd.read_csv("./temp.csv")
prcAll = r2_values.values.T

y_1 = prcAll[0][:179]
y_2 = prcAll[0][179:]

print(prcAll.shape)
for i in range(179):

    print(i, y_1[i], y_2[i])


x = list(range(71, 250))
# y_3 = three[2]
fig, ax = plt.subplots()
ax.plot(x, y_1, color="red", marker="o", markersize=2.5)
ax2 = ax.twinx()
ax2.plot(x, y_2, color="blue", marker="o", markersize=2.5)
plt.show()



