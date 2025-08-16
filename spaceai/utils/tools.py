import os
import zipfile
from typing import (
    Optional,
    List,
    Tuple,
)

from tqdm import tqdm
import numpy as np

# ``requests`` is a convenient dependency for downloading files but it is not
# strictly required for the rest of the project.  Some minimal testing
# environments (including the execution sandbox used for this repository) do
# not have it installed and network access may be restricted.  We therefore
# attempt to import it optionally and fall back to ``urllib`` when unavailable.
try:  # pragma: no cover - handled via tests
    import requests  # type: ignore[import-untyped]
except ModuleNotFoundError:  # pragma: no cover - exercised when requests missing
    requests = None  # type: ignore[assignment]
    import urllib.request as urllib_request

def download_and_extract_zip(url: str, extract_to: str, cleanup: bool = False):
    filename = download_file(url)
    extract_zip(filename, extract_to, cleanup)

def download_file(url: str, to: Optional[str] = None):
    if to is None:
        local_filename = url.split("/")[-1]
    else:
        local_filename = to
    if requests is not None:
        # Preferred implementation using ``requests`` when available.
        with requests.get(url, stream=True) as r:  # type: ignore[union-attr]
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            with open(local_filename, "wb") as f, tqdm(
                desc=local_filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=block_size):
                    f.write(chunk)
                    bar.update(len(chunk))
    else:  # Fallback using urllib when ``requests`` is missing.
        with urllib_request.urlopen(url) as response, open(local_filename, "wb") as f:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024
            with tqdm(
                desc=local_filename,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
    return local_filename

def extract_zip(filename: str, extract_to: str, cleanup: bool = False):
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    if cleanup:
        os.remove(filename)

def _compute_valid_gaps(total_len: int, invalid_masks: List[Tuple[int, int]], mask_len: int) -> List[Tuple[int, int]]:
    """Return sorted list of gaps (start, end) able to fit `mask_len`."""
    blocks = sorted((max(0, a), min(total_len, b)) for a, b in invalid_masks if a < b)
    gaps: List[Tuple[int, int]] = []
    prev = 0
    for a, b in blocks:
        if prev < a:
            gaps.append((prev, a))
        prev = max(prev, b)
    if prev < total_len:
        gaps.append((prev, total_len))
    return [g for g in gaps if g[1] - g[0] >= mask_len]

def all_smart_masks(
    total_len: int,
    invalid_masks: List[Tuple[int,int]],
    mask_len: int
) -> List[Tuple[int,int]]:
    """
    Genera TUTTE le maschere di lunghezza `mask_len` che 
    starebbero “a passo” all’interno di ciascun gap valido.
    """
    valid_gaps = _compute_valid_gaps(total_len, invalid_masks, mask_len)
    masks: List[Tuple[int,int]] = []
    for g0, g1 in valid_gaps:
        # quante maschere “piene” ci stanno in questo gap?
        count = (g1 - g0) // mask_len
        # partiamo sempre dall'estremità destra
        for i in range(count):
            start = g1 - mask_len*(i+1)
            masks.append((start, start + mask_len))
    return masks

def sample_smart_masks(
    total_len: int,
    invalid_masks: List[Tuple[int,int]],
    mask_len: int,
    n_masks: int,
    rng: np.random.Generator = None
) -> List[Tuple[int,int]]:
    """
    Estrae `n_masks` **a caso**, senza sovrapposizioni, fra
    tutte quelle generate da `all_smart_masks`.
    """
    if rng is None:
        rng = np.random.default_rng()

    candidates = all_smart_masks(total_len, invalid_masks, mask_len)
    if len(candidates) == 0:
        raise ValueError("No candidate masks available")
    # if there are fewer candidates than required, fall back to sampling with
    # replacement. This allows callers to request more masks than the distinct
    # non‑overlapping intervals available.
    replace = len(candidates) < n_masks
    chosen = rng.choice(len(candidates), size=n_masks, replace=replace)
    return [candidates[i] for i in chosen]

def make_smart_masks(
    total_len: int,
    invalid_masks: List[Tuple[int, int]],
    mask_len: int,
    n_masks: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, int]]:
    """Generate `n_masks` non‑overlapping contiguous masks.

    Parameters
    ----------
    total_len : int
        Length of the full signal.
    invalid_masks : list[(start,end)]
        Pre‑existing prohibited intervals.
    mask_len : int
        Desired length of each new mask.
    n_masks : int
        How many masks to generate.
    rng : np.random.Generator, optional

    Returns
    -------
    masks : list[(start,end)]
        Generated masks, length == n_masks.
    """
    if rng is None:
        rng = np.random.default_rng()

    valid_gaps = _compute_valid_gaps(total_len, invalid_masks, mask_len)
    if not valid_gaps:
        raise ValueError("No gap large enough for mask_len")

    # Start cursor at the *end* of the right‑most gap
    gap_idx = len(valid_gaps) - 1
    cursor = valid_gaps[gap_idx][1] - mask_len

    masks: List[Tuple[int, int]] = []
    while len(masks) < n_masks:
        g0, g1 = valid_gaps[gap_idx]
        if cursor < g0:
            # exhausted current gap -> move to previous gap or random restart
            gap_idx -= 1
            if gap_idx < 0:
                # restart from a random gap with random offset
                gap_idx = int(rng.integers(0, len(valid_gaps)))
                g0, g1 = valid_gaps[gap_idx]
                offset_max = g1 - g0 - mask_len
                cursor = g0 + int(rng.integers(0, offset_max + 1))
            else:
                # position cursor at end of the new gap
                cursor = valid_gaps[gap_idx][1] - mask_len
            continue  # re‑evaluate with new cursor/gap

        # create mask and step left
        start = cursor
        end = start + mask_len
        masks.append((start, end))
        cursor -= mask_len

    return masks

def generate_kfold_masks(
    total_len: int,
    perc_eval1: float,
) -> List[Tuple[int, int]]:
    """
    Divide [0, total_len) in n_splits contigue di lunghezza mask_len,
    dove mask_len = floor(perc_eval1 * total_len). Gli ultimi rimasti
    (se total_len non è divisibile) vengono ignorati.
    """
    mask_len = int(perc_eval1 * total_len)
    if mask_len < 1:
        raise ValueError("perc_eval1 troppo piccolo per generare una maschera")
    n_splits = total_len // mask_len
    masks: List[Tuple[int,int]] = []
    for i in range(n_splits):
        start = i * mask_len
        end = start + mask_len
        masks.append((start, end))
    return masks
