import itertools
from pyalgotrade.barfeed import yahoofeed
from pyalgotrade.optimizer import server

# Iterate over a range of parameters to find optimal values
def parameters_generator():
    instrument = ["f"] #ford stock
    entrySMA = range(150, 251)
    exitSMA = range(5, 16)
    rsiPeriod = range(2, 11)
    overBoughtThreshold = range(75, 96)
    overSoldThreshold = range(5, 26)
    return itertools.product(instrument, entrySMA, exitSMA, rsiPeriod, overBoughtThreshold, overSoldThreshold)

# The if __name__ == '__main__' part is necessary if running on Windows.
if __name__ == '__main__':
    # Load the feed from the CSV files.  Using Ford motor co. stock prices from sample years
    feed = yahoofeed.Feed()
    feed.addBarsFromCSV("f", "f-2007data.csv")
    feed.addBarsFromCSV("f", "f-2009data.csv")
    feed.addBarsFromCSV("f", "f-2014data.csv")

    # Run the server.
    server.serve(feed, parameters_generator(), "localhost", 5000)
