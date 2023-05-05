from typing import Dict, List

import datetime
import math

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)


def get_data(
    symbols: List[str],
    start_date: datetime.datetime,
    end_date: datetime.datetime,
) -> Dict:
    """Extract from yahoo the time-series data for the given symbols.

    Args:
        symbols (List[str]): List of symbols to get the data.
        start_date (datetime.datetime): Start date of the dataset.
        end_date (datetime.datetime): End date of the dataset.

    Returns:
        Dict: Dictionaries with all the datasets for all symbols.
    """
    data = {
        symbol: yf.download(symbol, start=start_date, end=end_date)
        for symbol in symbols
    }
    return data


def calculate_arithmetic_returns(time_series):
    """
    Calculate arithmetic returns from the given time series.

    Parameters:
        time_series (pandas.Series): Time series data.

    Returns:
        pandas.Series: Arithmetic returns with date as the index.
    """
    returns = time_series.pct_change().dropna()
    return returns


def plot_lines(
    data,
    labels=None,
    colors=None,
    title=None,
    xlabel=None,
    ylabel=None,
    figsize=(10, 6),
    legend_loc="best",
):
    """Plot lines on same graph."""
    fig, ax = plt.subplots(figsize=figsize)

    if colors and len(colors) != len(data):
        raise ValueError("Number of colors must match the number of data series")

    for i, d in enumerate(data):
        if colors:
            ax.plot(d, color=colors[i])
        else:
            ax.plot(d)

    if labels:
        ax.legend(labels, loc=legend_loc)

    if title:
        ax.set_title(title)

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    plt.show()


def plot_two_series(series1, series2, title, label1, label2, color1, color2):
    """Plot two series on same graph."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(series1.index, series1.values, color=color1, label=label1)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(label1, color=color1)
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()

    ax2.plot(series2.index, series2.values, color=color2, label=label2)
    ax2.set_ylabel(label2, color=color2)
    ax2.tick_params(axis="y")

    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_time_series(time_series, title, color="black"):
    """Plot a time series."""
    plt.figure(figsize=(10, 6))
    plt.plot(time_series.index, time_series.values, color=color)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()


def plot_returns(returns, title, color="black"):
    """
    Plot the given arithmetic returns.

    Parameters:
        returns (pandas.Series): Arithmetic returns data.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(returns.index, returns.values, color=color)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Returns")
    plt.grid(True)
    plt.show()


def plot_histogram(returns, title, color="black", bins=50):
    """
    Plot a histogram of the given arithmetic returns.

    Parameters:
        returns (pandas.Series): Arithmetic returns data.
        title (str): Title of the plot.
        color (str, optional): Color of the histogram bars. Default is "black".
        bins (int, optional): Number of bins to divide the data. Default is 50.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(returns.values, bins=bins, color=color, edgecolor="white", linewidth=1)
    plt.title(title)
    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def get_geometric_brownian_motion(
    P0: float, mu: float, sigma: float, T: float, dt: float
) -> np.ndarray:
    """Simulate a geometric brownian motion.

    Args:
        P0 (float): P0 parameter
        mu (float): µ parameter
        sigma (float): sigma parameter
        T (float): T parameter
        dt (float): Time step

    Returns:
        np.ndarray: _description_
    """
    N = int(T / dt)
    t = np.linspace(0, T, N)
    dB = np.sqrt(dt) * np.random.normal(0, 1, N)
    Bt = np.cumsum(dB)
    Pt = P0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * Bt)
    return Pt


def get_arithmetic_return(prices: np.ndarray) -> np.ndarray:
    """Get arithmetic retuns.

    Args:
        returns (np.ndarray): Numpy array containing price data.

    Returns:
        np.ndarray: Arithmetic returns
    """
    return np.diff(prices) / prices[:-1]


def get_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate the Exponential Moving Average (EMA) for a given price series and period.

    Args:
        prices (np.ndarray): Numpy array containing price data.
        period (int): Period size.

    Returns:
        np.ndarray: _description_
    """
    ema = prices.ewm(span=period, adjust=False).mean()
    return ema


def get_moving_standard_deviation(prices, period):
    """Calculate the moving standard deviation for a given period.

    Args:
        prices (_type_): _description_
        period (_type_): _description_

    Returns:
        _type_: _description_
    """
    return prices.rolling(window=period).std()


def get_xks(short_emas, long_emas):
    """Compute for each set of short / long EMA the difference xk.

    Args:
        short_emas (_type_): _description_
        long_emas (_type_): _description_

    Returns:
        _type_: _description_
    """
    x = {}
    for i, (short_ema, long_ema) in enumerate(zip(short_emas, long_emas)):
        x[f"x_{i+1}"] = short_ema - long_ema
    return x


def get_normalization(x, period: int, prices=None):
    """Normalize values using the moving standard deviation.

    Args:
        x: Values to be normalized.
        period: Period used to calculate the moving standard deviation.
        prices: Prices to be normalized if given.

    Returns:
        Normalized series.
    """
    if prices is not None:
        moving_standard_deviation = get_moving_standard_deviation(prices, period)
    else:
        moving_standard_deviation = get_moving_standard_deviation(x, period)
    return x / moving_standard_deviation


def get_signal_uk(zk: float) -> float:
    """Using the formula given in the paper, calculate Uk.

    Args:
        zk (float): Zk value

    Returns:
        float: Uk
    """
    return (zk * math.exp(-(zk**2) / 4)) / (math.sqrt(2) * math.exp(-1 / 2))


def weighted_sum(values: list, weights: list) -> float:
    """
    Calculate the weighted sum of values using the given weights.

    Args:
        values (list): A list of values.
        weights (list): A list of weights corresponding to the values.

    Returns:
        float: The weighted sum of the values.
    """
    if len(values) != len(weights):
        raise ValueError("The lengths of values and weights must be equal.")
    return np.dot(values, weights)
