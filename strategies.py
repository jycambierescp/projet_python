import backtrader as bt
import numpy as np

from data import get_signal_uk, weighted_sum

from constants import SHORT_EMA_PERIODS, LONG_EMA_PERIODS


class Data(bt.feeds.PandasData):
    """Data class to convert Yahoo data to Backtrader Data"""

    params = (
        ("datetime", None),
        ("open", -1),
        ("high", -1),
        ("low", -1),
        ("close", -1),
        ("volume", -1),
    )


class BaseStrategy(bt.Strategy):
    """
    Base strategy class that contains methods to compute the strategy's signal.
    """

    params = (
        ("short_emas", SHORT_EMA_PERIODS),
        ("long_emas", LONG_EMA_PERIODS),
        ("three_months_std_dev_period", 63),
        ("one_year_std_dev_period", 252),
        ("investment_per_position", 1000),
    )

    def __init__(self):
        self.signals = {}

    def setup_emas_differences(self):
        """
        Compute EMA differences for each data.
        """
        self.emas_differences = {}
        for data in self.datas:
            ema_difference_list = []
            for short_period, long_period in zip(
                self.params.short_emas, self.params.long_emas
            ):
                short_ema = bt.indicators.ExponentialMovingAverage(
                    data.close, period=short_period
                )
                long_ema = bt.indicators.ExponentialMovingAverage(
                    data.close, period=long_period
                )
                ema_difference = short_ema - long_ema
                ema_difference_list.append(ema_difference)
            self.emas_differences[data._name] = ema_difference_list

    def setup_standard_deviations(self):
        """
        Compute standard deviations for each data
        """
        self.standard_deviations = {}
        for data in self.datas:
            self.standard_deviations[data._name] = {}
            self.standard_deviations[data._name][
                "three_months"
            ] = bt.indicators.StandardDeviation(
                data.close, period=self.params.three_months_std_dev_period
            )

            self.standard_deviations[data._name]["one_year"] = {}
            for i, ema_difference in enumerate(self.emas_differences[data._name]):
                self.standard_deviations[data._name]["one_year"][
                    i
                ] = bt.indicators.StandardDeviation(
                    ema_difference
                    / self.standard_deviations[data._name]["three_months"],
                    period=self.params.one_year_std_dev_period,
                )

    def compute_signals(self):
        """
        Compute trading signal for each data using ema differences and standard deviations.
        """
        weights = [1 / len(self.params.short_emas)] * len(self.params.short_emas)
        for data in self.datas:
            uk_values = []
            for i, ema_difference in enumerate(self.emas_differences[data._name]):
                yk = (
                    ema_difference
                    / self.standard_deviations[data._name]["three_months"][0]
                )
                zk = yk / self.standard_deviations[data._name]["one_year"][i][0]
                uk = get_signal_uk(zk)
                uk_values.append(uk)
            self.signals[data._name] = weighted_sum(uk_values, weights)


class TimeSeriesPortfolioStrategy(BaseStrategy):
    """
    Time series portfolio strategy
    """

    def __init__(self):
        super().__init__()
        self.setup_emas_differences()
        self.setup_standard_deviations()

    def next(self):
        """
        This portfolio is rebalanced at the start of each day. On each date, we invest in all the currencies
        according to the value of the signal divided by n (number of currencies in the portfolio).

        For a signal of 1 for example, we buy 1/n units. For a signal of -1, we sell 1/n units.
        """
        n = len(self.datas)
        self.compute_signals()

        for data in self.datas:
            self.close(data)
            position_size = (
                self.params.investment_per_position
                * self.signals[data._name]
                / n
                / data.close[0]
            )
            if self.signals[data._name] > 0:
                self.buy(data=data, size=position_size)
            elif self.signals[data._name] < 0:
                self.sell(data=data, size=position_size)


class CrossSectionalStrategy(BaseStrategy):
    """
    Cross-Sectional portfolio strategy
    """

    def __init__(self):
        super().__init__()
        self.setup_emas_differences()
        self.setup_standard_deviations()

    def next(self):
        """
        This portfolio is rebalanced at the start of each day.
        However, it requires having at least 6 currencies in the portfolio.
        Depending on the signals returned, we buy the three currencies with the highest signal
        and sell the three currencies with the weakest signal.
        We always buy or sell exactly 1/6 units of USD of each currency.
        """
        self.compute_signals()

        for data in self.datas:
            self.close(data)

        sorted_signals = sorted(self.signals.items(), key=lambda x: x[1], reverse=True)

        top_3 = [item[0] for item in sorted_signals[:3]]
        bottom_3 = [item[0] for item in sorted_signals[-3:]]

        position_size = self.params.investment_per_position * 1 / 6 / data.close[0]

        if data._name in top_3:
            self.buy(data=data, size=position_size)

        elif data._name in bottom_3:
            self.sell(data=data, size=position_size)
