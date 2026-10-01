"""Tests for the 64-element apodization mask helpers."""
from __future__ import annotations

import numpy as np
import pytest

from openlifu_verification import apodization as apo


# ---------------------------------------------------------------------------
# Representation conversions
# ---------------------------------------------------------------------------
class TestParseHex:
    def test_all_off(self):
        assert apo.parse_hex("0x0000000000000000") == 0

    def test_all_on(self):
        assert apo.parse_hex("0xFFFFFFFFFFFFFFFF") == (1 << 64) - 1

    def test_single_element_1(self):
        assert apo.parse_hex("0x0000000000000001") == 1

    def test_no_prefix(self):
        assert apo.parse_hex("FFFF") == 0xFFFF

    def test_lower_case(self):
        assert apo.parse_hex("0xdeadbeef") == 0xDEADBEEF

    def test_underscores(self):
        assert apo.parse_hex("0xFFFF_FFFF_FFFF_FFFF") == (1 << 64) - 1

    def test_whitespace_stripped(self):
        assert apo.parse_hex("  0x1  ") == 1

    def test_too_long_rejects(self):
        # 17 F's = 68 bits, doesn't fit.
        with pytest.raises(ValueError, match="longer than 16"):
            apo.parse_hex("0xFFFFFFFFFFFFFFFFF")

    def test_non_hex_rejects(self):
        with pytest.raises(ValueError, match="not a hexadecimal"):
            apo.parse_hex("0xGG")

    def test_empty_rejects(self):
        with pytest.raises(ValueError, match="empty"):
            apo.parse_hex("0x")

    def test_wrong_type_rejects(self):
        with pytest.raises(TypeError):
            apo.parse_hex(123)  # type: ignore[arg-type]


class TestFormatHex:
    def test_canonical_padding(self):
        assert apo.format_hex(1) == "0x0000000000000001"

    def test_upper_case(self):
        assert apo.format_hex(0xabcdef) == "0x0000000000ABCDEF"

    def test_all_on(self):
        assert apo.format_hex((1 << 64) - 1) == apo.HEX_ALL_ON

    def test_negative_rejects(self):
        with pytest.raises(ValueError):
            apo.format_hex(-1)

    def test_overflow_rejects(self):
        with pytest.raises(ValueError):
            apo.format_hex(1 << 64)


class TestRoundTrip:
    @pytest.mark.parametrize("hex_code", [
        "0x0000000000000000",
        "0x0000000000000001",
        "0x8000000000000000",
        "0xFFFFFFFFFFFFFFFF",
        "0xDEADBEEFCAFEBABE",
    ])
    def test_parse_format(self, hex_code):
        assert apo.format_hex(apo.parse_hex(hex_code)) == hex_code


class TestMaskArrayRoundTrip:
    def test_all_on_array(self):
        arr = apo.mask_to_array(apo.HEX_ALL_ON)
        assert arr.shape == (64,)
        assert np.all(arr == 1.0)

    def test_all_off_array(self):
        arr = apo.mask_to_array(apo.HEX_ALL_OFF)
        assert np.all(arr == 0.0)

    def test_single_element(self):
        arr = apo.mask_to_array("0x0000000000000001")
        assert arr[0] == 1.0
        assert arr[1:].sum() == 0.0

    def test_short_array(self):
        arr = apo.mask_to_array("0x0F", n=8)
        assert arr.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]

    def test_array_to_mask(self):
        a = np.zeros(64)
        a[0] = 1.0
        a[63] = 1.0
        m = apo.array_to_mask(a)
        assert m == (1 << 0) | (1 << 63)

    def test_roundtrip_random(self):
        # Build the 64-bit sample from two 32-bit halves; NumPy's
        # ``Generator.integers`` doesn't handle a 64-bit unsigned
        # upper bound in the default int64 dtype.
        rng = np.random.default_rng(42)
        for _ in range(20):
            lo = int(rng.integers(0, 1 << 32))
            hi = int(rng.integers(0, 1 << 32))
            m = (hi << 32) | lo
            recovered = apo.array_to_mask(apo.mask_to_array(m))
            assert recovered == m


class TestNormalize:
    def test_int(self):
        assert apo.normalize_to_int(0xABCD) == 0xABCD

    def test_str(self):
        assert apo.normalize_to_int("0xABCD") == 0xABCD

    def test_ndarray(self):
        a = np.zeros(64)
        a[3] = 1.0
        assert apo.normalize_to_int(a) == (1 << 3)

    def test_list(self):
        assert apo.normalize_to_int([1, 0, 1]) == 0b101

    def test_bool_rejects(self):
        with pytest.raises(TypeError):
            apo.normalize_to_int(True)


class TestCountActive:
    def test_none(self):
        assert apo.count_active(apo.HEX_ALL_OFF) == 0

    def test_all(self):
        assert apo.count_active(apo.HEX_ALL_ON) == 64

    def test_single(self):
        assert apo.count_active("0x0000000000000001") == 1


# ---------------------------------------------------------------------------
# Index-based builders
# ---------------------------------------------------------------------------
class TestSingleElement:
    def test_element_1(self):
        assert apo.single_element(1) == "0x0000000000000001"

    def test_element_64(self):
        assert apo.single_element(64) == "0x8000000000000000"

    def test_zero_rejects(self):
        with pytest.raises(ValueError):
            apo.single_element(0)

    def test_65_rejects(self):
        with pytest.raises(ValueError):
            apo.single_element(65)


class TestElements:
    def test_empty(self):
        assert apo.elements([]) == apo.HEX_ALL_OFF

    def test_first_and_last(self):
        assert apo.elements([1, 64]) == "0x8000000000000001"

    def test_duplicates_ok(self):
        # OR'ing an already-on bit is a no-op.
        assert apo.elements([1, 1, 1]) == apo.single_element(1)

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            apo.elements([1, 65])


class TestAllElements:
    def test_default(self):
        assert apo.all_elements() == apo.HEX_ALL_ON

    def test_partial(self):
        assert apo.all_elements(4) == "0x000000000000000F"


class TestInvert:
    def test_all_on_to_all_off(self):
        assert apo.invert(apo.HEX_ALL_ON) == apo.HEX_ALL_OFF

    def test_all_off_to_all_on(self):
        assert apo.invert(apo.HEX_ALL_OFF) == apo.HEX_ALL_ON

    def test_partial_domain(self):
        # Only invert within first 4 bits.
        assert apo.invert("0x0000000000000005", n=4) == "0x000000000000000A"


# ---------------------------------------------------------------------------
# Position-based builders
# ---------------------------------------------------------------------------
@pytest.fixture
def grid_8x8() -> np.ndarray:
    """An 8x8 element grid centred at (0,0) with 5 mm pitch.

    Element 1 sits at (-17.5, -17.5, 0); indices go x-fastest,
    i.e. index = row * 8 + col (0-based)."""
    pts = []
    for row in range(8):
        for col in range(8):
            pts.append([col * 5.0 - 17.5, row * 5.0 - 17.5, 0.0])
    return np.array(pts, dtype=float)


class TestElementAt:
    def test_center(self, grid_8x8):
        # (0, 0) is between elements; nearest is element idx 27 or 28
        # (0-based) which is 1-based 28 or 29. Just verify it's one
        # of them and the mask has exactly one bit set.
        mask = apo.element_at(grid_8x8, 0.0, 0.0)
        assert apo.count_active(mask) == 1

    def test_bottom_left(self, grid_8x8):
        # Element 1 is at (-17.5, -17.5).
        mask = apo.element_at(grid_8x8, -17.5, -17.5)
        assert mask == apo.single_element(1)

    def test_max_distance_rejects(self, grid_8x8):
        with pytest.raises(ValueError, match="exceeds max_distance_mm"):
            apo.element_at(grid_8x8, 100.0, 0.0, max_distance_mm=5.0)

    def test_max_distance_allows(self, grid_8x8):
        # Within tolerance.
        mask = apo.element_at(grid_8x8, -17.5, -17.5, max_distance_mm=1.0)
        assert mask == apo.single_element(1)


class TestElementDisk:
    def test_all_in(self, grid_8x8):
        mask = apo.element_disk(grid_8x8, 0.0, 0.0, r=1000.0)
        assert apo.count_active(mask) == 64

    def test_none_in(self, grid_8x8):
        mask = apo.element_disk(grid_8x8, 100.0, 100.0, r=1.0)
        assert mask == apo.HEX_ALL_OFF

    def test_negative_radius(self, grid_8x8):
        with pytest.raises(ValueError):
            apo.element_disk(grid_8x8, 0.0, 0.0, r=-1.0)

    def test_boundary_inclusive(self, grid_8x8):
        # Element 1 at (-17.5, -17.5); distance from (-15, -15) is sqrt(2)*2.5.
        # Set r exactly to that distance; inclusive=True should include it.
        r = np.sqrt(2) * 2.5
        mask_in = apo.element_disk(grid_8x8, -15.0, -15.0, r=r,
                                   inclusive=True)
        mask_out = apo.element_disk(grid_8x8, -15.0, -15.0, r=r * 0.999,
                                    inclusive=True)
        assert apo.count_active(mask_in) > apo.count_active(mask_out)

    def test_small_disk(self, grid_8x8):
        # r < half-pitch centred on element 1 -> only element 1.
        mask = apo.element_disk(grid_8x8, -17.5, -17.5, r=1.0)
        assert mask == apo.single_element(1)


class TestElementRect:
    def test_full(self, grid_8x8):
        mask = apo.element_rect(grid_8x8, -100, 100, -100, 100)
        assert apo.count_active(mask) == 64

    def test_first_row(self, grid_8x8):
        # y in [-17.5, -17.5] captures the first row (8 elements).
        mask = apo.element_rect(grid_8x8, -100, 100, -17.5, -17.5)
        assert apo.count_active(mask) == 8
        # Those are elements 1..8.
        assert mask == apo.elements(range(1, 9))

    def test_invalid_rectangle(self, grid_8x8):
        with pytest.raises(ValueError):
            apo.element_rect(grid_8x8, 5, -5, 0, 0)


# ---------------------------------------------------------------------------
# Duck-typing: real transducer file
# ---------------------------------------------------------------------------
class TestWithTransducer:
    """Verify the positional helpers work with the real
    :class:`_TransducerData` loaded from a JSON file."""

    def test_1x_all(self):
        # Import lazily so pytest collection doesn't touch numpy-heavy
        # SDK imports.
        from openlifu_verification.verificationtank import _TransducerData
        path = "transducers/openlifu_1x_evt1.json"
        arr = _TransducerData.from_file(path)
        arr.sort_by_pin()

        mask = apo.element_disk(arr, 0.0, 0.0, r=1000.0)
        assert apo.count_active(mask) == arr.numelements()

    def test_1x_center(self):
        from openlifu_verification.verificationtank import _TransducerData
        path = "transducers/openlifu_1x_evt1.json"
        arr = _TransducerData.from_file(path)
        arr.sort_by_pin()

        mask = apo.element_at(arr, 0.0, 0.0, max_distance_mm=10.0)
        # Exactly one element selected.
        assert apo.count_active(mask) == 1
