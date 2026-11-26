import pytest

from animal_shelter import data


def test_convert_camel_case():
    assert data.convert_camel_case("CamelCase") == "camel_case"
    assert data.convert_camel_case("CamelCASE") == "camel_case"
    assert data.convert_camel_case("camel-case") != "camel_case"
    assert data.convert_camel_case("camel_case") == "camel_case"
    assert data.convert_camel_case("camel case") != "camel_case"


@pytest.mark.parametrize("value", [123, None, ["CamelCase"], b"CamelCase"])
def test_convert_camel_case_non_str_raises(value):
    with pytest.raises(TypeError):
        data.convert_camel_case(value)


@pytest.mark.filterwarnings("error")
def test_load_data_fills_fully_empty_column(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "ID,Name,DateTime,OutcomeSubtype\n"
        "1,Ivo,2014-02-12 18:22:00,\n"
        "2,,2013-10-13 12:44:00,\n"
    )

    df = data.load_data(path)

    assert list(df["outcome_subtype"]) == ["Unknown", "Unknown"]
    assert list(df["name"]) == ["Ivo", "Unknown"]
    assert df["id"].dtype == "int64"
    assert df["date_time"].dtype.kind == "M"
