"""ezcal - Easy Calculation.

One-command first-principles workflows.  The default engine is
Quantum ESPRESSO; the engine layer is pluggable so that VASP or an MLIP
(SevenNet) can be swapped in without touching the CLI or the workflows.
"""

__version__ = "0.1.0"

from ezcal.config import Config, load_config  # noqa: E402,F401

__all__ = ["Config", "load_config", "__version__"]
