import warnings


class WConditionNumberWarning(UserWarning):
    """Warning for W matrix with high condition number."""
    pass


warnings.filterwarnings(
    "once",
    category=WConditionNumberWarning
)
