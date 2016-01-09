"""
===================
Isotonic Regression on Apple Options Chain
===================

"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state
from urllib2 import urlopen
#Scraping minute data from yahoo finance because i'm poor as fuck 
optionsUrl = 'https://finance.yahoo.com/q/op?s=AAPL+Options'
optionsPage = urlopen(optionsUrl)
soup = BeautifulSoup(optionsPage, 'lxml')
#Generate a stock options table using nested list comprehension
optionsTable = [
    [x.text for x in y.parent.contents]
    for y in soup.findAll('td', attrs={'class': 'yfnc_h', 'nowrap': ''})
]
print optionsTable
"""
n = 100
x = optionsTable
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(n,)) + 50. * np.log(1 + len(x))

###############################################################################
# Fit IsotonicRegression and LinearRegression models

ir = IsotonicRegression()

y_ = ir.fit_transform(x, y)

lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # x needs to be 2d for LinearRegression

###############################################################################
# plot result

segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidths(0.5 * np.ones(n))

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, y_, 'g.-', markersize=12)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.gca().add_collection(lc)
plt.legend(('Data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
plt.show()
