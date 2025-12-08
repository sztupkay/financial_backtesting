import pandas as pd
import pytest

from financial_backtesting import simulate_trailing_stop_np

_SAMPLE_DATA = {
    "Date": pd.date_range(start="2023-01-01", periods=5, freq="D"),
    "High": [101, 95, 110, 115, 120],
    "Low": [99, 90, 81, 110, 115],
    "Close": [100, 103, 108, 113, 118],
    "all_time_high": [101, 101, 110, 115, 120]
}

def test_simulate_trailing_stop_no_transaction():
    # Create a simple DataFrame to test
    df = pd.DataFrame(_SAMPLE_DATA)

    # Test with some parameters
    result = simulate_trailing_stop_np(0.9, 10, 0.1, df)
    assert result == 18  

def test_simulate_trailing_stop_stop_loss_no_reentry():
    # Create a simple DataFrame to test
    df = pd.DataFrame(_SAMPLE_DATA)

    # Test with some parameters
    result = simulate_trailing_stop_np(0.1, 10, 0.5, df)
    expected_return = (909 - 1000) / 1000  # Bought at 100, sold at 94.5 (stop loss), re-bought at 99, sold at 115
    assert result == expected_return * 100

def test_simulate_trailing_stop_stop_loss_reentry_by_price():
    # Create a simple DataFrame to test
    df = pd.DataFrame(_SAMPLE_DATA)

    # Test with some parameters
    result = simulate_trailing_stop_np(0.1, 10, 0.1, df)
    assert result == 31.111111111111106

def test_simulate_trailing_stop_stop_loss_reentry_by_day():
    # Create a simple DataFrame to test
    df = pd.DataFrame(_SAMPLE_DATA)

    # Test with some parameters
    result = simulate_trailing_stop_np(0.1, 2, 0.5, df)
    assert result == -4.655999999999995


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))