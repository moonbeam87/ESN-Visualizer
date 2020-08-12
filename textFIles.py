import numpy as np
import pandas as pd


def make_textfiles(series):
    for name in series:
        clone = pd.read_csv(
            "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="
            + name
            + "&apikey=WCXVE7BAD668SJHL&datatype=csv&outputsize=full"
        )
        clone.astype({"close": "float64"}).dtypes
        columns = clone["close"]
        columns = columns[::-1]
        fileName = name + ".txt"
        np.savetxt(fileName, columns.values, fmt="%f")


names = ["AAPL", "GOOGL", "FB", "IBM", "AMZN"]
make_textfiles(names)
