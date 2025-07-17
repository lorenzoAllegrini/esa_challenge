"""spaceai package initialization."""

import numpy as np

# ---------------------------------------------------------------------------
# Compatibility shims
# ---------------------------------------------------------------------------
# NumPy removed several aliases such as ``np.int`` in version 2.0.  Some
# dependencies (e.g. scikit-optimize) still rely on these names.  To avoid
# runtime failures when the project is used with newer NumPy releases, we
# restore the alias if it is missing.  This mirrors the workaround used in the
# test suite and is safe because it simply points ``np.int`` to the builtin
# ``int`` type.
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

__version__ = "0.0.1"
