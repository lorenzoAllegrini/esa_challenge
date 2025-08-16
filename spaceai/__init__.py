"""spaceai package initialization."""

import numpy as np

# ---------------------------------------------------------------------------
# Compatibility shims
# ---------------------------------------------------------------------------
# NumPy removed several aliases such as ``np.int`` and ``np.bool`` in version
# 2.0. Some dependencies (e.g. scikit-optimize or older SciPy releases) still
# rely on these names. To avoid runtime failures when the project is used with
# newer NumPy releases, we restore the aliases if they are missing. This mirrors
# the workaround used in the test suite and is safe because it simply points the
# aliases to the corresponding built-in types.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = bool  # type: ignore[attr-defined]

__version__ = "0.0.1"
