import pandas as pd
from pandas.testing import assert_series_equal

from animal_shelter import features


def test_check_has_name():
    s = pd.Series(["Ivo", "Henk", "unknown"])
    result = features._check_has_name(s)
    expected = pd.Series([True, True, False])
    assert_series_equal(result, expected)
