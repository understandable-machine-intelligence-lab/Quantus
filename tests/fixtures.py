import pytest

data_cases = {"a_subset_0.5_s": None,
              "s_subset_0.5_a": None,
              "a_equal_s": None,
              "a_geq_s": None,
              "a_empty_s": None,
              }

@pytest.fixture(scope="session", params=[])
def data(request):
    #request.cls.num1 = 10
