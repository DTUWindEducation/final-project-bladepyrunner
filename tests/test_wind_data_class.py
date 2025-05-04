import pytest
import pandas as pd
from datetime import datetime
from your_module_path.wind_data_class import WindData  # Update this import path

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'Time': [datetime(2021, 1, 1, 0), datetime(2021, 1, 1, 1)],
        'Wind Speed': [5.0, 6.2],
        'Power Output': [150.0, 200.0]
    })

def test_init_wind_data(sample_dataframe):
    wd = WindData(sample_dataframe)
    assert isinstance(wd.df, pd.DataFrame)
    assert 'Time' in wd.df.columns

def test_get_column_existing(sample_dataframe):
    wd = WindData(sample_dataframe)
    wind_speeds = wd.get_column('Wind Speed')
    assert list(wind_speeds) == [5.0, 6.2]

def test_get_column_missing(sample_dataframe):
    wd = WindData(sample_dataframe)
    with pytest.raises(KeyError):
        wd.get_column('Nonexistent Column')