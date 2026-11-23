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
