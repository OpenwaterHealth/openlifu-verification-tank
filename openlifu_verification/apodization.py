"""64-element apodization mask helpers.

The OpenLIFU verification hardware drives up to 64 array elements; a
convenient way to specify which elements are active on a given pulse
is a 64-bit mask, one bit per element. Following the convention the
user asked for:

* Bit ``0`` (LSB) is element **1** (1-based, matching the ``pin``
  numbering in the transducer JSON files).
* ``0x0000000000000001`` fires element 1 only.
* ``0xFFFFFFFFFFFFFFFF`` fires all 64 elements.

The functions here all speak the same three interchangeable
representations:

* **hex string** - a 16-char (or 18-char with ``0x``) string; what a
  human types.
* **integer**    - the same value as a Python ``int``; what bitwise
  ops want.
* **numpy array** - a ``(numelements,)`` float array of 0/1 that can
  be dropped straight into
  :meth:`VerificationTank.configure_lifu`'s ``apodizations``
  parameter.

Positional helpers (:func:`element_at`, :func:`element_disk`) take a
transducer array-like (anything with ``get_positions(units="mm")``
that returns a shape ``(N, 3)`` array of element centers, as in
:class:`openlifu_verification.verificationtank._TransducerData`) and
return a hex mask, so a script can write::

    from openlifu_verification import apodization as apo
    mask = apo.element_disk(ver.arr, x0=0.0, y0=0.0, r=5.0)
    ver.apply_pulse(apodization=mask)

None of these helpers touch hardware; they're pure functions ready
to import from tests or scripts.
"""
from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np


# Hardware limit: the TX7332 chain drives up to 64 channels.
NUM_ELEMENTS: int = 64

# The bit mask that fits exactly ``NUM_ELEMENTS`` bits.
_ALL_ON_MASK: int = (1 << NUM_ELEMENTS) - 1

# Handy constants scripts can import instead of typing the literal.
HEX_ALL_OFF: str = "0x0000000000000000"
HEX_ALL_ON: str = "0xFFFFFFFFFFFFFFFF"

# Anything that behaves like an apodization mask when we don't want
# to care which representation the caller used.
MaskLike = Union[int, str, np.ndarray, Iterable[float]]


# ----------------------------------------------------------------------
# Representation conversions
# ----------------------------------------------------------------------
def _validate_mask_int(value: int) -> int:
    """Ensure ``value`` fits in ``NUM_ELEMENTS`` bits.

    Rejects negative integers and anything above :data:`_ALL_ON_MASK`
    so a typo like ``0xFFFFFFFFFFFFFFFFF`` (17 F's) is caught at the
    boundary instead of silently truncating."""
    if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
        raise TypeError(f"expected int mask, got {type(value).__name__}")
    value = int(value)
    if value < 0:
        raise ValueError(f"mask must be non-negative, got {value:#x}")
    if value > _ALL_ON_MASK:
        raise ValueError(
            f"mask 0x{value:X} exceeds {NUM_ELEMENTS}-bit range "
            f"(max {_ALL_ON_MASK:#018X})"
        )
    return value


def parse_hex(hex_code: str) -> int:
    """Parse a hex apodization string into an integer.

    Accepts optional ``0x`` / ``0X`` prefix, underscores as thousands
    separators (Python-style), and any capitalisation. Leading zeros
    are optional; strings up to 16 hex digits (64 bits) are valid.
    Raises :class:`ValueError` on anything else so operator typos
    surface immediately."""
    if not isinstance(hex_code, str):
        raise TypeError(
            f"expected hex string, got {type(hex_code).__name__}"
        )
    text = hex_code.strip().replace("_", "")
    if text.lower().startswith("0x"):
        text = text[2:]
    if not text:
        raise ValueError(f"empty hex code {hex_code!r}")
    if len(text) > 16:
        raise ValueError(
            f"hex code {hex_code!r} is longer than 16 digits "
            f"(64 bits)"
        )
    try:
        value = int(text, 16)
    except ValueError as exc:
        raise ValueError(
            f"not a hexadecimal integer: {hex_code!r}"
        ) from exc
    return _validate_mask_int(value)


def format_hex(mask: int) -> str:
    """Format ``mask`` as a canonical zero-padded 16-digit
    ``0x…`` string, upper case. Round-trips with :func:`parse_hex`."""
    return f"0x{_validate_mask_int(mask):016X}"


def mask_to_array(mask: MaskLike, n: int = NUM_ELEMENTS) -> np.ndarray:
    """Return a ``(n,)`` float array of 0.0 / 1.0.

    Accepts anything :func:`normalize_to_int` accepts. ``n`` lets a
    caller ask for a shorter array (e.g. a 32-element transducer);
    the low ``n`` bits are returned in element order (element 1 at
    index 0)."""
    if n <= 0 or n > NUM_ELEMENTS:
        raise ValueError(f"n must be in 1..{NUM_ELEMENTS}, got {n}")
    m = normalize_to_int(mask)
    out = np.zeros(n, dtype=float)
    for i in range(n):
        if (m >> i) & 1:
            out[i] = 1.0
    return out


def array_to_mask(arr: Iterable[float], *, threshold: float = 0.5) -> int:
    """Convert a 0/1 element array into an int mask.

    Any array entry ``>= threshold`` is treated as "on". Extra
    elements past :data:`NUM_ELEMENTS` are rejected so the caller
    can't silently overflow the 64-bit budget."""
    a = np.asarray(list(arr), dtype=float).reshape(-1)
    if a.size > NUM_ELEMENTS:
        raise ValueError(
            f"apodization array has {a.size} entries, max is "
            f"{NUM_ELEMENTS}"
        )
    value = 0
    for i, v in enumerate(a):
        if float(v) >= threshold:
            value |= 1 << i
    return _validate_mask_int(value)


def normalize_to_int(mask: MaskLike) -> int:
    """Coerce any accepted representation into an int mask.

    Dispatch: ``int`` → validated; ``str`` → :func:`parse_hex`;
    ``np.ndarray`` or any other iterable → :func:`array_to_mask`.
    """
    if isinstance(mask, bool):
        raise TypeError("bool is not a valid apodization mask")
    if isinstance(mask, (int, np.integer)):
        return _validate_mask_int(int(mask))
    if isinstance(mask, str):
        return parse_hex(mask)
    if isinstance(mask, np.ndarray) or hasattr(mask, "__iter__"):
        return array_to_mask(mask)
    raise TypeError(
        f"cannot interpret {type(mask).__name__} as an apodization mask"
    )


def count_active(mask: MaskLike) -> int:
    """Return the number of active elements in ``mask``."""
    return bin(normalize_to_int(mask)).count("1")


# ----------------------------------------------------------------------
# Index-based mask builders
# ----------------------------------------------------------------------
def single_element(index: int) -> str:
    """Hex mask that lights up exactly one 1-based element index.

    ``single_element(1) == '0x0000000000000001'``.
    """
    idx = int(index)
    if idx < 1 or idx > NUM_ELEMENTS:
        raise ValueError(
            f"element index must be in 1..{NUM_ELEMENTS}, got {idx}"
        )
    return format_hex(1 << (idx - 1))


def elements(indices: Iterable[int]) -> str:
    """Hex mask covering all 1-based element indices in ``indices``.

    Duplicate indices are silently OR'd together; that's harmless
    (already-on bits stay on) but a caller that mixes 0-based and
    1-based numbers would still get a wrong-but-plausible mask, so
    the input is bounds-checked strictly."""
    value = 0
    for i in indices:
        idx = int(i)
        if idx < 1 or idx > NUM_ELEMENTS:
            raise ValueError(
                f"element index must be in 1..{NUM_ELEMENTS}, got {idx}"
            )
        value |= 1 << (idx - 1)
    return format_hex(value)


def all_elements(n: int = NUM_ELEMENTS) -> str:
    """Hex mask for the first ``n`` elements.

    ``all_elements()`` returns :data:`HEX_ALL_ON`."""
    if n <= 0 or n > NUM_ELEMENTS:
        raise ValueError(f"n must be in 1..{NUM_ELEMENTS}, got {n}")
    return format_hex((1 << n) - 1)


def invert(mask: MaskLike, n: int = NUM_ELEMENTS) -> str:
    """Return the mask that fires the *complementary* elements.

    Only the first ``n`` bits are toggled (the rest stay 0) so the
    result stays within the transducer's element budget."""
    if n <= 0 or n > NUM_ELEMENTS:
        raise ValueError(f"n must be in 1..{NUM_ELEMENTS}, got {n}")
    m = normalize_to_int(mask)
    return format_hex((~m) & ((1 << n) - 1))


# ----------------------------------------------------------------------
# Position-based mask builders
# ----------------------------------------------------------------------
def _positions_2d(array) -> np.ndarray:
    """Extract the (x, y) columns from ``array.get_positions("mm")``.

    ``array`` is duck-typed so tests can pass a NumPy array directly
    instead of a full ``_TransducerData``."""
    if isinstance(array, np.ndarray):
        pos = np.asarray(array, dtype=float)
    else:
        pos = np.asarray(array.get_positions(units="mm"), dtype=float)
    if pos.ndim != 2 or pos.shape[1] < 2:
        raise ValueError(
            f"expected (N, >=2) positions array, got shape {pos.shape}"
        )
    if pos.shape[0] > NUM_ELEMENTS:
        raise ValueError(
            f"transducer has {pos.shape[0]} elements, max is "
            f"{NUM_ELEMENTS}"
        )
    return pos[:, :2]


def element_at(array, x0: float, y0: float,
               *, max_distance_mm: Optional[float] = None) -> str:
    """Hex mask for the single element nearest ``(x0, y0)``.

    ``max_distance_mm`` (when given) rejects picks farther than that
    distance so a typo like ``x0=100`` on a 25-mm-radius aperture
    doesn't silently select the rim element."""
    pos = _positions_2d(array)
    target = np.asarray([float(x0), float(y0)], dtype=float)
    d = np.linalg.norm(pos - target, axis=1)
    idx0 = int(np.argmin(d))
    if max_distance_mm is not None and d[idx0] > float(max_distance_mm):
        raise ValueError(
            f"nearest element is {d[idx0]:.3f} mm from ({x0}, {y0}), "
            f"which exceeds max_distance_mm={max_distance_mm}"
        )
    # Convert from 0-based array index to 1-based element index.
    return single_element(idx0 + 1)


def element_disk(array, x0: float, y0: float, r: float,
                 *, inclusive: bool = True) -> str:
    """Hex mask for every element whose center lies within radius
    ``r`` of ``(x0, y0)``.

    ``inclusive=True`` (default) includes elements exactly on the
    boundary; set to ``False`` to require strict inequality. An
    empty result (no elements within ``r``) is a valid mask
    (:data:`HEX_ALL_OFF`) - the caller decides whether that's an
    error."""
    if r < 0:
        raise ValueError(f"radius must be non-negative, got {r}")
    pos = _positions_2d(array)
    target = np.asarray([float(x0), float(y0)], dtype=float)
    d = np.linalg.norm(pos - target, axis=1)
    if inclusive:
        hits = np.where(d <= float(r))[0]
    else:
        hits = np.where(d < float(r))[0]
    # array indices are 0-based; hex mask is 1-based.
    return elements(int(i) + 1 for i in hits)


def element_rect(array, x_min: float, x_max: float,
                 y_min: float, y_max: float) -> str:
    """Hex mask for every element whose center lies inside the
    axis-aligned rectangle ``[x_min, x_max] x [y_min, y_max]``.

    Useful for annular / row / column patterns. Boundaries are
    inclusive."""
    if x_min > x_max or y_min > y_max:
        raise ValueError(
            f"invalid rectangle: x=[{x_min}, {x_max}], "
            f"y=[{y_min}, {y_max}]"
        )
    pos = _positions_2d(array)
    hits = np.where(
        (pos[:, 0] >= float(x_min)) & (pos[:, 0] <= float(x_max)) &
        (pos[:, 1] >= float(y_min)) & (pos[:, 1] <= float(y_max))
    )[0]
    return elements(int(i) + 1 for i in hits)
