import warnings


class WConditionNumberWarning(UserWarning):
    """Warning for W matrix with high condition number."""
    pass


warnings.filterwarnings(
    "once",
    category=WConditionNumberWarning
)

# cupyx.scipy.signal defines internal CUDA kernels with @cupyx.jit.rawkernel,
# which fires this FutureWarning at import time whenever the cupy backend is
# loaded.  The warning is upstream noise; suppress it globally.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="cupyx.jit.rawkernel is experimental.*",
)
