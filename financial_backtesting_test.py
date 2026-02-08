import pandas as pd
import pytest

from financial_backtesting import simulate_trailing_stop_np, Position, Reentry

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


class TestPosition:
    """Tests for the Position class."""

    def test_position_initialization(self):
        """Test that position initializes with correct capital."""
        initial_capital = 1000.0
        pos = Position(initial_capital)
        assert pos.cash == initial_capital
        assert pos.shares == 0.0
        assert not pos.is_in_shares()

    def test_position_buy(self):
        """Test buying shares converts cash to shares."""
        initial_capital = 1000.0
        pos = Position(initial_capital)
        price = 100.0
        date = pd.Timestamp("2023-01-01")

        pos.buy(price, date)

        assert pos.shares == 10.0  # 1000 / 100
        assert pos.cash == 0.0
        assert pos.is_in_shares()

    def test_position_buy_raises_error_when_already_holding_shares(self):
        """Test that buying when already holding shares raises an error."""
        initial_capital = 1000.0
        pos = Position(initial_capital)
        date = pd.Timestamp("2023-01-01")

        pos.buy(100.0, date)

        with pytest.raises(ValueError, match="Cannot buy: already holding shares"):
            pos.buy(100.0, date)

    def test_position_sell(self):
        """Test selling shares converts shares to cash."""
        initial_capital = 1000.0
        pos = Position(initial_capital)
        buy_price = 100.0
        sell_price = 150.0
        date = pd.Timestamp("2023-01-01")

        pos.buy(buy_price, date)
        pos.sell(sell_price, date)

        assert pos.shares == 0.0
        assert pos.cash == 1500.0  # 10 shares * 150
        assert not pos.is_in_shares()

    def test_position_sell_raises_error_when_already_holding_cash(self):
        """Test that selling when already holding cash raises an error."""
        pos = Position(1000.0)

        with pytest.raises(ValueError, match="Cannot sell: already holding cash"):
            pos.sell(100.0, pd.Timestamp("2023-01-01"))

    def test_position_get_value_with_shares(self):
        """Test get_value calculates correct value when holding shares."""
        initial_capital = 1000.0
        pos = Position(initial_capital)
        pos.buy(100.0, pd.Timestamp("2023-01-01"))

        # 10 shares at 150 = 1500
        value = pos.get_value(150.0)
        assert value == 1500.0

    def test_position_get_value_with_cash(self):
        """Test get_value returns cash value when holding cash."""
        initial_capital = 1000.0
        pos = Position(initial_capital)

        value = pos.get_value(100.0)
        assert value == 1000.0  # No shares, just cash

    def test_position_get_value_after_sell(self):
        """Test get_value after selling shares."""
        pos = Position(1000.0)
        pos.buy(100.0, pd.Timestamp("2023-01-01"))
        pos.sell(150.0, pd.Timestamp("2023-01-02"))

        # After selling: 1500 cash, price doesn't matter
        value = pos.get_value(200.0)
        assert value == 1500.0

    def test_position_buy_and_sell_cycle(self):
        """Test a complete buy-sell cycle with profit."""
        initial_capital = 1000.0
        pos = Position(initial_capital)

        # Buy at 100
        pos.buy(100.0, pd.Timestamp("2023-01-01"))
        assert pos.shares == 10.0
        assert pos.cash == 0.0

        # Sell at 150 (50% gain)
        pos.sell(150.0, pd.Timestamp("2023-01-02"))
        assert pos.shares == 0.0
        assert pos.cash == 1500.0
        assert not pos.is_in_shares()

    def test_position_buy_sell_buy_cycle(self):
        """Test multiple buy-sell cycles."""
        pos = Position(1000.0)

        # First cycle: buy at 100, sell at 150
        pos.buy(100.0, pd.Timestamp("2023-01-01"))
        assert pos.shares == 10.0
        pos.sell(150.0, pd.Timestamp("2023-01-02"))
        assert pos.cash == 1500.0

        # Second cycle: buy at 150, sell at 180
        pos.buy(150.0, pd.Timestamp("2023-01-03"))
        assert pos.shares == 10.0  # 1500 / 150
        pos.sell(180.0, pd.Timestamp("2023-01-04"))
        assert pos.cash == 1800.0

    def test_position_get_value_fractional_shares(self):
        """Test get_value with fractional shares."""
        pos = Position(1000.0)
        pos.buy(300.0, pd.Timestamp("2023-01-01"))

        # 1000 / 300 = 3.333... shares
        assert pos.shares == pytest.approx(1000.0 / 300.0)
        
        # Value at price 400 = 3.333... * 400 = 1333.33...
        value = pos.get_value(400.0)
        assert value == pytest.approx(1000.0 / 300.0 * 400.0)

    def test_position_is_in_shares_consistency(self):
        """Test is_in_shares returns correct state throughout lifecycle."""
        pos = Position(1000.0)

        # Initially in cash
        assert not pos.is_in_shares()

        # After buy, in shares
        pos.buy(100.0, pd.Timestamp("2023-01-01"))
        assert pos.is_in_shares()

        # After sell, in cash
        pos.sell(150.0, pd.Timestamp("2023-01-02"))
        assert not pos.is_in_shares()


class TestReentry:
    """Tests for the Reentry class."""

    def test_reentry_initialization(self):
        """Test that Reentry initializes with correct price and date."""
        price = 100.0
        date = pd.Timestamp("2023-01-15")
        reentry = Reentry(price=price, date=date)
        
        assert reentry.price == price
        assert reentry.date == date

    def test_should_reenter_by_price_drop(self):
        """Test re-entry triggered when price drops to reentry price."""
        reentry = Reentry(price=90.0, date=pd.Timestamp("2023-01-15"))
        
        # Low price equals reentry price
        result = reentry.should_reenter(low_price=90.0, high_price=95.0, date=pd.Timestamp("2023-01-10"))
        assert result == 90.0

    def test_should_reenter_by_price_drop_below(self):
        """Test re-entry triggered when price drops below reentry price."""
        reentry = Reentry(price=90.0, date=pd.Timestamp("2023-01-15"))
        
        # Low price is below reentry price
        result = reentry.should_reenter(low_price=85.0, high_price=95.0, date=pd.Timestamp("2023-01-10"))
        assert result == 90.0

    def test_should_reenter_by_date_reached(self):
        """Test re-entry triggered when reentry date is reached."""
        reentry = Reentry(price=90.0, date=pd.Timestamp("2023-01-15"))
        
        # Date exactly equals reentry date, low price not at reentry price
        result = reentry.should_reenter(low_price=100.0, high_price=110.0, date=pd.Timestamp("2023-01-15"))
        expected = (100.0 + 110.0) / 2  # midpoint of low and high
        assert result == expected

    def test_should_reenter_by_date_passed(self):
        """Test re-entry triggered when reentry date is passed."""
        reentry = Reentry(price=90.0, date=pd.Timestamp("2023-01-15"))
        
        # Date after reentry date, low price not at reentry price
        result = reentry.should_reenter(low_price=100.0, high_price=110.0, date=pd.Timestamp("2023-01-20"))
        expected = (100.0 + 110.0) / 2
        assert result == expected

    def test_should_reenter_no_trigger(self):
        """Test no re-entry when neither price nor date condition is met."""
        reentry = Reentry(price=90.0, date=pd.Timestamp("2023-01-15"))
        
        # Price above reentry price and date before reentry date
        result = reentry.should_reenter(low_price=100.0, high_price=110.0, date=pd.Timestamp("2023-01-10"))
        assert result is None

    def test_should_reenter_price_takes_precedence(self):
        """Test that price-based re-entry triggers even if date condition also met."""
        reentry = Reentry(price=90.0, date=pd.Timestamp("2023-01-15"))
        
        # Both price and date conditions met - price should be returned
        result = reentry.should_reenter(low_price=85.0, high_price=110.0, date=pd.Timestamp("2023-01-20"))
        assert result == 90.0  # Price-based re-entry takes precedence

    def test_should_reenter_date_based_uses_midpoint(self):
        """Test that date-based re-entry returns midpoint of low and high."""
        reentry = Reentry(price=90.0, date=pd.Timestamp("2023-01-15"))
        
        # Date condition met but price condition not, should return midpoint
        result = reentry.should_reenter(low_price=110.0, high_price=130.0, date=pd.Timestamp("2023-01-15"))
        expected = (110.0 + 130.0) / 2
        assert result == expected
        assert result == 120.0

    def test_should_reenter_with_equal_low_and_high(self):
        """Test date-based re-entry when low and high are equal."""
        reentry = Reentry(price=90.0, date=pd.Timestamp("2023-01-15"))
        
        # When low and high are the same, midpoint should be the same value
        result = reentry.should_reenter(low_price=100.0, high_price=100.0, date=pd.Timestamp("2023-01-15"))
        assert result == 100.0

    def test_should_reenter_low_just_above_threshold(self):
        """Test no re-entry when low price is just above reentry price."""
        reentry = Reentry(price=90.0, date=pd.Timestamp("2023-01-15"))
        
        # Low is just above reentry price and date before reentry date
        result = reentry.should_reenter(low_price=90.01, high_price=95.0, date=pd.Timestamp("2023-01-10"))
        assert result is None

    def test_should_reenter_with_negative_prices(self):
        """Test re-entry logic works with unusual price scenarios."""
        reentry = Reentry(price=1.0, date=pd.Timestamp("2023-01-15"))
        
        # Reentry with very small price
        result = reentry.should_reenter(low_price=0.5, high_price=1.5, date=pd.Timestamp("2023-01-10"))
        assert result == 1.0

    def test_should_reenter_date_comparison_precision(self):
        """Test date comparison with timestamps that have time components."""
        reentry = Reentry(price=90.0, date=pd.Timestamp("2023-01-15 10:00:00"))
        
        # Date before reentry time
        result = reentry.should_reenter(low_price=100.0, high_price=110.0, date=pd.Timestamp("2023-01-15 09:59:59"))
        assert result is None
        
        # Date at reentry time
        result = reentry.should_reenter(low_price=100.0, high_price=110.0, date=pd.Timestamp("2023-01-15 10:00:00"))
        assert result == 105.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))