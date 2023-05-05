import backtrader as bt

from strategies import Data, TimeSeriesPortfolioStrategy, CrossSectionalStrategy


def run_backtest(
    dataset, portfolio, three_months_std_dev_period, one_year_std_dev_period
):
    """Run backtest on given dataset with given portfolio strategy."""
    if portfolio not in ["time_series", "cross_sectional"]:
        raise ValueError("Portfolio should be time_series or cross_sectional.")

    cerebro = bt.Cerebro()

    for symbol in dataset:
        data_feed = Data(dataname=dataset[symbol])
        cerebro.adddata(data_feed, name=symbol)

    if portfolio == "time_series":
        cerebro.addstrategy(
            TimeSeriesPortfolioStrategy,
            three_months_std_dev_period=three_months_std_dev_period,
            one_year_std_dev_period=one_year_std_dev_period,
        )
    else:
        cerebro.addstrategy(CrossSectionalStrategy)

    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Years, _name="annual_return"
    )
    cerebro.addanalyzer(
        bt.analyzers.SharpeRatio,
        timeframe=bt.TimeFrame.Years,
        _name="annual_sharpe_ratio",
    )
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annual_std_dev")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    results = cerebro.run()

    return results, cerebro
