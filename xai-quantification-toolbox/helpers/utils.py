from typing import Union, Optional


def check_if_fitted(m) -> Optional[bool]:
    """Checks if a measure is fitted by the presence of """
    if not hasattr(m, 'fit'):
        raise TypeError(f"{m} is not an instance." )
    return True

"""
    if not check_if_fitted:
     print(f"This {Measure.name} is instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
"""