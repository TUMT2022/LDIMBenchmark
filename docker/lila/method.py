import os
import logging
from ldimbenchmark import (
    FileBasedMethodRunner,
)

from ldimbenchmark.methods.lila import LILA

# read log level from environment variable
logLevel = os.getenv("LOG_LEVEL", "INFO")

numeric_level = getattr(logging, logLevel, None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % logLevel)

logging.basicConfig(level=numeric_level, handlers=[logging.StreamHandler()])
logging.getLogger().setLevel(numeric_level)

if __name__ == "__main__":
    runner = FileBasedMethodRunner(LILA())
    runner.run()
