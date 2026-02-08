from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import itertools
from dataclasses import dataclass

_HISTORICAL_CSV_PATH = "/Users/sztupkay/Documents/scripts/financial_backtesting/monthly_qqq.csv"


@dataclass
class Reentry:
    """Represents a re-entry condition for buying back into a position.
    
    Re-entry occurs if either:
    - The current price drops to or below the 'price' field, OR
    - The current date reaches or passes the 'date' field.
    """
    price: float
    date: pd.Timestamp

    def should_reenter(self, low_price: float, high_price: float, date: pd.Timestamp) -> Optional[float]:
        """Check if re-entry condition is met based on price or date."""
        if low_price <= self.price:
            return self.price
        elif date >= self.date:
            return (low_price + high_price) / 2
        return None


class ReentryFactory:
    """Factory for creating Reentry conditions based on strategy parameters."""
    
    def __init__(self, reentry_drop_ratio: float, reentry_wait_days: int):
        self.reentry_drop_ratio = reentry_drop_ratio
        self.reentry_wait_days = reentry_wait_days
    
    def create(self, sale_price: float, sale_date: pd.Timestamp) -> Reentry:
        """Create a Reentry condition from a sale price and date."""
        reentry_price = sale_price * (1.0 - self.reentry_drop_ratio)
        reentry_date = sale_date + pd.to_timedelta(self.reentry_wait_days, unit='D')
        return Reentry(price=reentry_price, date=reentry_date)    

class Position:
    """Represents a trading position holding either cash or shares."""
    
    def __init__(self, initial_capital: float):
        self.shares = 0.0
        self.cash = initial_capital
    
    def buy(self, price: float, date: pd.Timestamp) -> None:
        """Buy shares with all available cash at given price."""
        if self.shares > 0:
            raise ValueError("Cannot buy: already holding shares")
        amount = self.cash / price
        self.shares = amount
        # print(f"🟢 BOUGHT: {self.shares:.2f} shares @ {price:.2f} on {date.strftime('%Y-%m-%d')} ({self.cash:.2f}$)")
        self.cash = 0.0
    
    def sell(self, price: float, date: pd.Timestamp) -> None:
        """Sell all shares at given price for cash."""
        if self.cash > 0:
            raise ValueError("Cannot sell: already holding cash")
        self.cash = self.shares * price
        # print(f"🛑 SOLD: {self.shares:.2f} shares @ {price:.2f} on {date.strftime('%Y-%m-%d')} ({self.cash:.2f}$)")
        self.shares = 0.0
        
    def get_value(self, current_price: float) -> float:
        """Get total position value at current price."""
        return self.shares * current_price + self.cash
    
    def is_in_shares(self) -> bool:
        """Check if currently holding shares."""
        if self.shares > 0 and self.cash > 0:
            raise ValueError("Invalid state: position cannot hold both shares and cash")
        return self.shares > 0
# ----------------------------------------------------------------------
# 1. Backtester class
# ----------------------------------------------------------------------

class Backtester:
    """Represents a trading position (either in or out of the market)."""
    
    def __init__(self, initial_capital: float, historical_data: pd.DataFrame, stop_loss_ratio: float, reentry_factory: ReentryFactory):
        self.initial_capital = initial_capital
        self.stop_loss_ratio = stop_loss_ratio
        self.reentry_factory = reentry_factory
        self.reentry = None  # type: Reentry | None
        self.position = Position(initial_capital)
        self.position.buy((historical_data.iloc[0]['High'] + historical_data.iloc[0]['Low'])/2, historical_data.iloc[0]['Date'])
        self.historical_data = historical_data
        self.prev_high = historical_data.iloc[0]['High']
        self.day_index = 0

    def next_day(self):
        self.day_index += 1
        if self.day_index >= len(self.historical_data):
            return False  # No more data
        if self.position.is_in_shares():
            self.run_in_shares_day()
        else:
            self.run_in_cash_day()
        row = self.historical_data.iloc[self.day_index]
        self.prev_high = max(self.prev_high, row['High'])
        return True
    
    def run_in_shares_day(self):
        # Logic for when the position is currently in the market
        row = self.historical_data.iloc[self.day_index]
        stop_loss_limit_price = self.prev_high * (1.0 - self.stop_loss_ratio)
        if row['Low'] < stop_loss_limit_price:
            # 🛑 STOP LOSS: Sell the position
            # print(f"Stop loss triggered: Low {row['Low']:.2f} < prev_high {self.prev_high:.2f} with stop_loss_limit_price {stop_loss_limit_price:.2f}")
            sale_price = min(row['High'], stop_loss_limit_price)
            sale_date = row['Date']
            self.position.sell(sale_price, sale_date)

            self.reentry = self.reentry_factory.create(sale_price, sale_date)
        else:
            # 📈 HOLD: Remain in the position
            self.in_position = True        

    def run_in_cash_day(self):
        # Logic for when the position is currently out of the market
        row = self.historical_data.iloc[self.day_index]
        if self.reentry is None:
            raise ValueError("Reentry condition is not set.")
        reentry_price = self.reentry.should_reenter(row['Low'], row['High'], row['Date'])
        if reentry_price is not None:
            # Re-enter the position
            self.position.buy(reentry_price, row['Date'])
            self.reentry = None
            self.prev_high = row['High'] # Reset prev_high on time-based re-entry


# ----------------------------------------------------------------------
# 2. Strategy simulator
# ----------------------------------------------------------------------
# Sell when Low <= peak_high * (1 - X).
# After selling at Low, re-enter when either:
#   - Close <= sale_price * (1 - Z_fixed), OR
#   - Y days have passed since sale.
# All-in, integer shares, no transaction costs.

def simulate_trailing_stop_np(stop_loss_ratio: float, wait_days: int, reentry_drop_ratio: float, historical_data: pd.DataFrame) -> float:
    """Simulates a trailing stop loss strategy with reentry on additional drops.
    
    Backtests a trading strategy that uses a trailing stop loss to exit positions,
    with the ability to reenter the market when the price drops an additional amount
    after exiting. Starts with an initial capital of $1000 and processes historical
    price data day by day.
    
    Args:
        stop_loss_ratio (float): Trailing stop loss as a decimal fraction.
            For example, 0.13 represents a 13% trailing stop.
        wait_days (int): Number of days to wait before allowing reentry after
            a stop loss is triggered.
        reentry_drop_ratio (float): Additional price drop required to trigger reentry
            as a decimal fraction. For example, 0.14 represents a 14% additional drop.
        historical_data (pd.DataFrame): Historical OHLC price data with at least
            a 'Close' column. Data should be sorted chronologically.
    
    Returns:
        float: Return on capital (ROC) as a percentage. For example, 5.25 represents
            a 5.25% return on the initial $1000 capital.
    
    Raises:
        Exception: May raise exceptions from Backtester or ReentryFactory if invalid
            parameters are provided.
    
    Example:
        >>> roc = simulate_trailing_stop_np(0.13, 5, 0.14, historical_data)
        >>> print(f"Strategy returned {roc:.2f}%")
    """
    reentry_factory = ReentryFactory(reentry_drop_ratio, wait_days)
    backtester = Backtester(initial_capital=1000.0, historical_data=historical_data, stop_loss_ratio=stop_loss_ratio, reentry_factory=reentry_factory)

    while backtester.next_day():
        pass
    final_value = backtester.position.get_value(current_price=historical_data.iloc[-1]['Close'])
    final_roc = (final_value - 1000.0) / 1000.0
    final_roc_pct = final_roc * 100.0
    print(f"Final ROC with stop_loss_ratio={stop_loss_ratio:.2%}, wait_days={wait_days}, reentry_drop_ratio={reentry_drop_ratio:.2%}: {final_roc_pct:.2f}%")
    return final_roc * 100.0


# ----------------------------------------------------------------------
# 3. Parameter grid and heatmap generator
# ----------------------------------------------------------------------

def run_grid_and_plot(reentry_drop_ratio: float, threshold_roc: float, historical_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a grid of X and Y for given Z, compute ROC (%), and show heatmap
    where red < threshold_roc and green >= threshold_roc.

    Arguments:
        reentry_drop_ratio: fixed re-entry drop as fraction (e.g., 0.14 for 14%). Re-entry refers to buying
                 back into a position after having previously sold it, based on the price dropping
                 by this fraction from the sale price, or after a certain number of days have passed.
        threshold_roc: threshold return on capital (%) for color mapping
    """
    # X: 2%..30% in 2% steps, Y: 4..40 days in steps of 3
    stop_loss_ratio_values: list[float] = [x / 100 for x in range(2, 31, 2)]
    wait_days_values: list[int] = list(range(4, 41, 3))

    rows: list[dict[str, float]] = []
    for stop_loss_ratio, wait_days in itertools.product(stop_loss_ratio_values, wait_days_values):
        roc = simulate_trailing_stop_np(stop_loss_ratio, wait_days, reentry_drop_ratio, historical_data)
        rows.append({"stop_loss_ratio": stop_loss_ratio, "wait_days": wait_days, "ROC": roc})

    sens_df: pd.DataFrame = pd.DataFrame(rows)

    # Pivot to matrix: rows = X, columns = Y, values = ROC
    heat  = sens_df.pivot(index="stop_loss_ratio", columns="wait_days", values="ROC").sort_index()
    vals: np.ndarray = heat.values
    print(vals)

    # Build custom red/green color map around threshold_roc
    rgb: np.ndarray = np.zeros((vals.shape[0], vals.shape[1], 3))

    # Below threshold: shades of red
    below: np.ndarray = vals < threshold_roc
    if np.any(below):
        min_b: float = np.nanmin(vals[below])
        denom_b: float = (threshold_roc - min_b) if threshold_roc != min_b else 1.0
        t_b: np.ndarray = (vals - min_b) / denom_b
        t_b = np.clip(t_b, 0, 1)
        rgb[..., 0] += below * 1.0
        rgb[..., 1] += below * (1.0 - t_b)
        rgb[..., 2] += below * (1.0 - t_b)

    # Above or equal threshold: shades of green
    above: np.ndarray = vals >= threshold_roc
    if np.any(above):
        max_a: float = np.nanmax(vals[above])
        denom_a: float = (max_a - threshold_roc) if max_a != threshold_roc else 1.0
        t_a: np.ndarray = (vals - threshold_roc) / denom_a
        t_a = np.clip(t_a, 0, 1)
        rgb[..., 1] += above * 1.0
        rgb[..., 0] += above * (1.0 - t_a)
    # Plot heatmap
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(rgb, aspect="auto", origin="lower")

    # Add text annotations for each cell to display the ROC value
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            roc_value = vals[i, j]
            # Choose text color based on the background color for readability
            # If the value is below the threshold (reddish background), use white text.
            # Otherwise (greenish background), use black text.
            text_color = "white" if roc_value < threshold_roc else "black"
            ax.text(j, i, f"{roc_value:.0f}%",
                    ha="center", va="center", color=text_color, fontsize=8)


    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels([f"{x:.0%}" for x in heat.index])

    ax.set_xlabel("Y (wait days)")
    ax.set_ylabel("X (trailing stop)")
    ax.set_title(
        f"QQQ Trailing-Stop Sensitivity — ROC (%)\n"
        f"Re-entry drop fixed at {int(reentry_drop_ratio*100)}% (red < {threshold_roc}%, green ≥ {threshold_roc}%)"
    )

    plt.tight_layout()
    plt.show()

    return sens_df, heat

def prep_historical_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    if "Close/Last" in df.columns:
        df = df.rename(columns={"Close/Last": "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    # Calculate the all-time high observed up to each row's date.
    # The expanding creates an expanding window object, which allows calculations 
    # (like max(), mean(), sum()) over all preceding rows up to the current one.
    df["all_time_high"] = df["High"].expanding().max()
    df["pullback_ratio"] = (df["all_time_high"] - df["Low"]) / df["all_time_high"]
    return df

def get_buy_and_hold_roc(df: pd.DataFrame) -> float:
    initial_cost = 1000 #* df["Close"].iloc[0]
    shares = initial_cost / ((df["High"].iloc[0] + df["Low"].iloc[0]) / 2)
    buy_and_hold_final_value: float = shares * df["Close"].iloc[-1]
    buy_and_hold_roc: float = (buy_and_hold_final_value - initial_cost) / initial_cost * 100.0
    print(f"Buy and Hold Final Value: {buy_and_hold_final_value:.2f}, ROC: {buy_and_hold_roc:.2f}%")
    return buy_and_hold_roc

def backtesting():
    df = pd.read_csv(_HISTORICAL_CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    if "Close/Last" in df.columns:
        df = df.rename(columns={"Close/Last": "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    global high_arr, low_arr, close_arr, initial_cost
    high_arr = df["High"].to_numpy()
    low_arr = df["Low"].to_numpy()
    close_arr = df["Close"].to_numpy()
    # initial_cost = 1000 * close_arr[0]
    # buy_and_hold_final_value: float = 1000 * close_arr[-1]
    buy_and_hold_roc: float = get_buy_and_hold_roc(df)
    print(f"Buy and Hold ROC: {buy_and_hold_roc:.2f}%")
    run_grid_and_plot(reentry_drop_ratio=0.14, threshold_roc=buy_and_hold_roc, historical_data=prep_historical_data(_HISTORICAL_CSV_PATH))


def main():
    backtesting()

if __name__ == "__main__":
    main()
