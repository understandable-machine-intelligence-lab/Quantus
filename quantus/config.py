from os import environ

# Use XLA compiler for TF functions, which promises to provide a noticeable speed-up
# for "a lot of small functions" case compared with regular Grappler. However, not supported on macOS.
USE_XLA = environ.get("USE_XLA", False)
TF_VECTORIZE_LOOPS = environ.get("TF_VECTORIZE_LOOPS", True)
