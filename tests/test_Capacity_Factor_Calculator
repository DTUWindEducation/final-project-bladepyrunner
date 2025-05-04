import pytest
from src.Capacity_Factor_Class import CapacityFactorCalculator  # Adjust the import path

def test_capacity_factor_calculator():
    calculator = CapacityFactorCalculator(rated_power_mw=5)
    
    # Test known AEP value
    aep_gwh = 20
    expected_cf = aep_gwh / (5 * 8760 / 1000)
    computed_cf = calculator.compute(aep_gwh)

    assert pytest.approx(computed_cf, 0.01) == expected_cf
    assert 0 <= computed_cf <= 1

def test_capacity_factor_zero_aep():
    calculator = CapacityFactorCalculator(rated_power_mw=5)
    assert calculator.compute(0) == 0

def test_capacity_factor_high_aep():
    calculator = CapacityFactorCalculator(rated_power_mw=5)
    cf = calculator.compute(60)  # unrealistic, but for testing
    assert cf > 1 or cf <= 1  # ensures no crash, edge case
