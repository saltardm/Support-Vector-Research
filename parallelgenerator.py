import itertools
from pyalgotrade.optimizer import local
from pyalgotrade.barfeed import yahoofeed
import rsi2
from pyalgotrade.tools import yahoofinance

def parameters_generator():
    instrument = ["f"]
    entrySMA = range(150, 251)
    exitSMA = range(5, 16)
    rsiPeriod = range(2, 11)
    overBoughtThreshold = range(75, 96)
    overSoldThreshold = range(5, 26)
    return itertools.product(instrument, entrySMA, exitSMA, rsiPeriod, overBoughtThreshold, overSoldThreshold)


# The if __name__ == '__main__' part is necessary if running on Windows.
if __name__ == '__main__':
    # Load the feed from the CSV files.
    yahoofinance.download_daily_bars('f', 2009, 'f-2009data.csv')
    feed = yahoofeed.Feed()
    feed.addBarsFromCSV("f", "f-2007data.csv")
    feed.addBarsFromCSV("f", "f-2009data.csv")
    feed.addBarsFromCSV("f", "f-2014data.csv")

    local.run(rsi2.RSI2, feed, parameters_generator())
