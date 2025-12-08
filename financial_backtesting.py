import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import itertools

_HISTORICAL_CSV_PATH = "/Users/sztupkay/Downloads/HistoricalData_1765146671158.csv"

# ----------------------------------------------------------------------
# 2. Strategy simulator
# ----------------------------------------------------------------------
# Sell when Low <= peak_high * (1 - X).
# After selling at Low, re-enter when either:
#   - Close <= sale_price * (1 - Z_fixed), OR
#   - Y days have passed since sale.
# All-in, integer shares, no transaction costs.

def simulate_trailing_stop_np(stop_loss_ratio: float, wait_days: int, reentry_drop_ratio: float, historical_data: pd.DataFrame) -> float:
    """
    stop_loss_ratio: trailing stop as fraction (e.g., 0.13 for 13%)
    wait_days: wait days (integer)
    reentry_drop_ratio: additional drop re-entry trigger as fraction (e.g., 0.14)
    """
    df = historical_data

    in_position = True
    current_number_of_shares: float = 1000/((df.iloc[0]['High'] + df.iloc[0]['Low']) / 2)
    current_cash = np.nan
    sale_price = np.nan
    reentry_price = np.nan
    sale_date = pd.NaT
    prev_high = df.iloc[0]['High']
    # Iterate through each day
    for index, row in df.iterrows():
        stop_loss_limit_price = prev_high * (1.0 - stop_loss_ratio)
        max_reentry_date = sale_date + pd.to_timedelta(wait_days, unit='D') if sale_date is not pd.NaT else pd.NaT

        if in_position and (not pd.isna(sale_price) or not pd.isna(sale_date) or not pd.isna(current_cash) or pd.isna(current_number_of_shares)):
            raise ValueError(f"Expected sale_price and sale_date to be NaN, and current_number_of_shares to be set, when in position. Got: sale_price={sale_price}, sale_date={sale_date}, current_number_of_shares={current_number_of_shares}")
        if not in_position and (pd.isna(sale_price) or pd.isna(sale_date) or pd.isna(current_cash) or not pd.isna(current_number_of_shares)):
            raise ValueError(
                f"Expected sale_price, sale_date, current_cash to be set, and "
                f"current_number_of_shares to be NaN, when not "
                f"in position. Got: sale_price={sale_price}, "
                f"sale_date={sale_date}, current_cash={current_cash}, "
                f"current_number_of_shares={current_number_of_shares}"
            )

        # --- LOGIC ---
        if in_position:
            # Check for Stop Loss Trigger
            if row['Low'] < stop_loss_limit_price:
                # 🛑 STOP LOSS: Sell the position
                in_position = False
                sale_price = min(row['High'], stop_loss_limit_price)
                sale_date = row['Date']
                current_cash = current_number_of_shares * sale_price
                # print(f"🛑 SOLD: {current_number_of_shares:.2f} shares @ {sale_price:.2f} on {row['Date'].strftime('%Y-%m-%d')}")
                reentry_price = sale_price * (1.0 - reentry_drop_ratio)
                current_number_of_shares = np.nan
            else:
                # 📈 HOLD: Remain in the position
                in_position = True        
        else: # Not in a position
            # Check for Buyback/Entry Signal
            if row['Low'] < reentry_price:
                # 🟢 ENTRY: Buy the stock
                in_position = True
                sale_price = np.nan
                sale_date = pd.NaT
                current_number_of_shares = current_cash / reentry_price
                # print(f"🟢 BOUGHT: {current_number_of_shares:.2f} shares @ {reentry_price:.2f} on {row['Date'].strftime('%Y-%m-%d')}")
                current_cash = np.nan
                reentry_price = np.nan
            elif row['Date'] >= max_reentry_date:
                # ⏰ TIME-BASED RE-ENTRY: Buy the stock
                in_position = True
                sale_price = np.nan
                sale_date = pd.NaT
                prev_high = row['High'] # Reset prev_high on time-based re-entry
                current_number_of_shares = current_cash / ((row['High'] + row['Low']) / 2)
                # print(f"🟢 BOUGHT: {current_number_of_shares:.2f} shares @ {(row['High'] + row['Low']) / 2:.2f} on {row['Date'].strftime('%Y-%m-%d')}")
                current_cash = np.nan
                reentry_price = np.nan
            else:
                # 😴 WAIT: Remain out of the position
                in_position = False
        prev_high = max(prev_high, row['High'])

    final_value: float = current_number_of_shares * df.iloc[-1]['Close'] if in_position else current_cash
    final_roc = (final_value - 1000.0) / 1000.0
    # if in_position:
    #     print(f"Final value (shares) with stop_loss_ratio={stop_loss_ratio:.2%}, wait_days={wait_days}, reentry_drop_ratio={reentry_drop_ratio:.2%}: {final_roc:.2%}")
    # else:
    #     print(f"Final value (cash) with stop_loss_ratio={stop_loss_ratio:.2%}, wait_days={wait_days}, reentry_drop_ratio={reentry_drop_ratio:.2%}: {final_roc:.2%}")
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
    initial_cost = 1000 * df["Close"].iloc[0]
    buy_and_hold_final_value: float = 1000 * df["Close"].iloc[-1]
    buy_and_hold_roc: float = (buy_and_hold_final_value - initial_cost) / initial_cost * 100.0
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
