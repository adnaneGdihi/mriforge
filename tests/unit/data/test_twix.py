"""Tests for the pure-Python Siemens TWIX ``.dat`` decoder (``mriforge.data.twix``).

TWIX is Siemens' bespoke raw-acquisition container; there is no public schema file,
so these tests build *byte-valid* synthetic fixtures (a VD multi-RAID file and a VB
single-measurement file) and assert the decoder reads them back correctly. The
decoder is the SSOT consumed by both the loader (``TwixStrategy``, full k-space) and
the profiler (``profile_twix``, header-only) — keeping it torch-free.

Header-only parsing (``parse_twix_header``) must work without materialising k-space
(the OOM lesson). k-space reading (``read_twix``) returns the imaging measurement's
*flat* acquisition list ``[n_acq, n_coil, n_samp]`` — like ``IsmrmrdStrategy`` — and
**raises** rather than guess a gridded layout or decode an unimplemented VB payload
(pitfall #9: no silent fallback).
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

from mriforge.data import twix

# --- synthetic TWIX byte builders ---------------------------------------------------

_ACQEND = 0x1  # aulEvalInfoMask bit 0 — last scan of a measurement


def _ascconv(
    *,
    field: float = 2.89362,
    seq: str = r"%SiemensSeq%\trufi",
    base: int = 128,
    pe: int = 128,
    contrasts: int = 1,
    tr_us: int = 5000,
    te_us: tuple[int, ...] = (2500,),
    flip: float = 40.0,
    coils: int = 4,
) -> str:
    """Build a minimal but realistic Siemens ASCCONV protocol block (+ XProtocol coil token)."""
    lines = [
        f'<ParamLong."iMaxNoOfRxChannels">  {{ {coils}  }}',
        "### ASCCONV BEGIN object=MrProtDataImpl@MrProtocolData version=41340007 ###",
        f"sProtConsistencyInfo.flNominalB0\t = \t{field}",
        f'tSequenceFileName\t = \t"{seq}"',
        f"sKSpace.lBaseResolution\t = \t{base}",
        f"sKSpace.lPhaseEncodingLines\t = \t{pe}",
        f"lContrasts\t = \t{contrasts}",
        f"alTR[0]\t = \t{tr_us}",
    ]
    for i, te in enumerate(te_us):
        lines.append(f"alTE[{i}]\t = \t{te}")
    lines.append(f"adFlipAngleDegree[0]\t = \t{flip}")
    lines.append("### ASCCONV END ###")
    return "\n".join(lines) + "\n"


def _scan_header(
    n_samp: int,
    n_chan: int,
    *,
    eval0: int = 0,
    lin: int = 0,
    ave: int = 0,
    sli: int = 0,
    eco: int = 0,
    set_: int = 0,
) -> bytes:
    """A 192-byte VD ``sScanHeader`` with the fields the reader uses.

    The ``sLoopCounter`` (14 uint16 from offset 52) carries Lin/Ave/Sli/…/Set; the
    reader surfaces all of them, so the builder can set the phase-cycle (``set_``),
    slice (``sli``), echo (``eco``) and average (``ave``) axes a real acquisition
    spreads across.
    """
    buf = bytearray(192)
    struct.pack_into("<I", buf, 0, 192 + n_chan * (32 + n_samp * 8))  # DMA length
    struct.pack_into("<I", buf, 40, eval0)  # aulEvalInfoMask[0]
    struct.pack_into("<H", buf, 48, n_samp)  # ushSamplesInScan
    struct.pack_into("<H", buf, 50, n_chan)  # ushUsedChannels
    struct.pack_into("<H", buf, 52, lin)  # sLoopCounter.Lin
    struct.pack_into("<H", buf, 54, ave)  # sLoopCounter.Ave (acquisition/average)
    struct.pack_into("<H", buf, 56, sli)  # sLoopCounter.Sli
    struct.pack_into("<H", buf, 60, eco)  # sLoopCounter.Eco
    struct.pack_into("<H", buf, 66, set_)  # sLoopCounter.Set (bSSFP phase-cycle)
    return bytes(buf)


def _channel(ch_id: int, samples: np.ndarray) -> bytes:
    """A 32-byte VD ``sChannelHeader`` + interleaved complex64 samples."""
    buf = bytearray(32)
    struct.pack_into("<H", buf, 24, ch_id)  # ushChannelId
    return bytes(buf) + np.asarray(samples, dtype=np.complex64).tobytes()


def _scan_block(data: np.ndarray, *, eval0: int = 0, lin: int = 0, **loop: int) -> bytes:
    """One full VD scan: header + every channel's header+samples. ``data`` is [n_chan, n_samp].

    Extra ``loop`` kwargs (``ave``/``sli``/``eco``/``set_``) set the other
    ``sLoopCounter`` fields.
    """
    n_chan, n_samp = data.shape
    parts = [_scan_header(n_samp, n_chan, eval0=eval0, lin=lin, **loop)]
    for c in range(n_chan):
        parts.append(_channel(c, data[c]))
    return b"".join(parts)


def _build_vd(path, measurements: list[dict]) -> None:
    """Write a VD multi-RAID ``.dat``.

    ``measurements`` is a list of ``{'protocol': str, 'acqs': [np.ndarray[n_chan,n_samp], ...]}``;
    each measurement gets an ACQEND terminator scan appended automatically.
    """
    n = len(measurements)
    entries_start = 8
    entries_len = n * 152
    cursor = entries_start + entries_len
    offsets, bodies = [], []
    for m in measurements:
        ptext = m["protocol"].encode("latin-1")
        hdr_len = 4 + len(ptext)
        # each acq is an ndarray, or a dict {"data": ndarray, "lin"/"set_"/"sli"/…: int}
        scan_bytes = b"".join(
            _scan_block(a["data"], **{k: v for k, v in a.items() if k != "data"})
            if isinstance(a, dict) else _scan_block(a)
            for a in m["acqs"]
        )
        scan_bytes += _scan_header(16, 1, eval0=_ACQEND)  # ACQEND terminator
        body = struct.pack("<I", hdr_len) + ptext + scan_bytes
        offsets.append(cursor)
        bodies.append(body)
        cursor += len(body)

    header = struct.pack("<II", 0, n)  # id=0, n_scans=n
    table = b"".join(
        struct.pack("<IIQQ64s64s", i + 1, 1, off, len(body), b"pat", b"prot")
        for i, (off, body) in enumerate(zip(offsets, bodies, strict=True))
    )
    path.write_bytes(header + table + b"".join(bodies))


def _build_vb(path, protocol: str) -> None:
    """Write a VB single-measurement ``.dat`` (header only — k-space unimplemented)."""
    ptext = protocol.encode("latin-1")
    hdr_len = 4 + len(ptext)
    path.write_bytes(struct.pack("<I", hdr_len) + ptext)


# --- ASCCONV parsing ----------------------------------------------------------------


def test_parse_ascconv_extracts_key_values() -> None:
    d = twix.parse_ascconv(_ascconv(field=0.55, base=96))
    assert d["sProtConsistencyInfo.flNominalB0"] == "0.55"
    assert d["sKSpace.lBaseResolution"] == "96"
    assert d["tSequenceFileName"] == r"%SiemensSeq%\trufi"  # quotes stripped


def test_parse_ascconv_without_block_is_empty() -> None:
    assert twix.parse_ascconv("no protocol here") == {}


# --- version detection --------------------------------------------------------------


def test_detect_version_vd(tmp_path) -> None:
    p = tmp_path / "vd.dat"
    _build_vd(p, [{"protocol": _ascconv(), "acqs": [np.zeros((4, 8))]}])
    with open(p, "rb") as fh:
        version, n_scans = twix.detect_twix_version(fh)
    assert version == "vd"
    assert n_scans == 1


def test_detect_version_vb(tmp_path) -> None:
    p = tmp_path / "vb.dat"
    _build_vb(p, _ascconv())
    with open(p, "rb") as fh:
        version, _ = twix.detect_twix_version(fh)
    assert version == "vb"


# --- header parsing (the high-leverage, header-only path) ---------------------------


def test_parse_twix_header_vd_fields(tmp_path) -> None:
    p = tmp_path / "img.dat"
    _build_vd(
        p,
        [{"protocol": _ascconv(field=2.89362, base=128, pe=120, contrasts=1,
                               tr_us=5000, te_us=(2500,), flip=40.0, coils=4),
          "acqs": [np.zeros((4, 64))]}],
    )
    hdr = twix.parse_twix_header(p)
    assert hdr.version == "vd"
    assert hdr.field_strength_t == pytest.approx(2.89362)
    assert hdr.sequence == "trufi"  # basename of tSequenceFileName, lowercased
    assert hdr.matrix == [128, 120]
    assert hdr.n_contrasts == 1
    assert hdr.te_ms == pytest.approx([2.5])  # 2500 us -> 2.5 ms
    assert hdr.tr_ms == pytest.approx([5.0])
    assert hdr.flip_deg == pytest.approx([40.0])
    assert hdr.n_coils == 4  # ushUsedChannels read from the first scan header


def test_parse_twix_header_uses_last_measurement(tmp_path) -> None:
    """Earlier measurements are noise/adjust; the imaging scan is the LAST one."""
    p = tmp_path / "multi.dat"
    _build_vd(
        p,
        [
            {"protocol": _ascconv(field=0.5, seq=r"%AdjSeq%\noise"),
             "acqs": [np.zeros((2, 8))]},
            {"protocol": _ascconv(field=2.89362, seq=r"%SiemensSeq%\trufi"),
             "acqs": [np.zeros((4, 16))]},
        ],
    )
    hdr = twix.parse_twix_header(p)
    assert hdr.field_strength_t == pytest.approx(2.89362)
    assert hdr.sequence == "trufi"


def test_parse_twix_header_vb_header_only(tmp_path) -> None:
    p = tmp_path / "vb.dat"
    _build_vb(p, _ascconv(field=1.494, base=256))
    hdr = twix.parse_twix_header(p)
    assert hdr.version == "vb"
    assert hdr.field_strength_t == pytest.approx(1.494)
    assert hdr.matrix == [256, 128]


def test_parse_twix_header_rejects_non_twix(tmp_path) -> None:
    p = tmp_path / "junk.dat"
    p.write_bytes(b"not a twix file, just some random bytes without a protocol")
    with pytest.raises(ValueError, match=r"ASCCONV|TWIX"):
        twix.parse_twix_header(p)


def test_parse_twix_header_rejects_truncated(tmp_path) -> None:
    p = tmp_path / "tiny.dat"
    p.write_bytes(b"\x00\x01")  # < 8 bytes
    with pytest.raises(ValueError):
        twix.parse_twix_header(p)


# --- k-space reading (flat acquisition list) ----------------------------------------


def test_read_twix_vd_kspace_shape_and_values(tmp_path) -> None:
    p = tmp_path / "img.dat"
    a0 = (np.arange(4 * 64).reshape(4, 64) + 1j * np.arange(4 * 64).reshape(4, 64)).astype(
        np.complex64
    )
    a1 = (a0 + 1000).astype(np.complex64)
    _build_vd(p, [{"protocol": _ascconv(coils=4), "acqs": [a0, a1]}])

    out = twix.read_twix(p)
    ks = out["kspace"]
    assert ks.shape == (2, 4, 64)  # [n_acq, n_coil, n_samp]
    assert np.iscomplexobj(ks)
    np.testing.assert_allclose(ks[0], a0)
    np.testing.assert_allclose(ks[1], a1)
    assert out["header"].n_coils == 4


def test_read_twix_surfaces_full_loop_counters(tmp_path) -> None:
    """sLoopCounter Set/Sli/Ave/Eco are surfaced so a phase-cycled multi-slice /
    dual-echo acquisition can be binned onto its true axes (not n_acq=n_pe)."""
    p = tmp_path / "pc.dat"
    base = np.ones((2, 8), dtype=np.complex64)
    # 2 phase cycles (Set) x 2 lines (Lin), slice 3, average 1, echo 0
    acqs = [
        {"data": base, "lin": 0, "set_": 0, "sli": 3, "ave": 1},
        {"data": base, "lin": 1, "set_": 0, "sli": 3, "ave": 1},
        {"data": base, "lin": 0, "set_": 1, "sli": 3, "ave": 1},
        {"data": base, "lin": 1, "set_": 1, "sli": 3, "ave": 1},
    ]
    _build_vd(p, [{"protocol": _ascconv(coils=2), "acqs": acqs}])

    out = twix.read_twix(p)
    lc = out["loop_counters"]
    assert set(lc) >= {"Lin", "Ave", "Sli", "Eco", "Set"}
    np.testing.assert_array_equal(lc["Lin"], [0, 1, 0, 1])
    np.testing.assert_array_equal(lc["Set"], [0, 0, 1, 1])  # the phase-cycle axis
    np.testing.assert_array_equal(lc["Sli"], [3, 3, 3, 3])
    np.testing.assert_array_equal(lc["Ave"], [1, 1, 1, 1])
    # back-compat alias still points at Lin
    np.testing.assert_array_equal(out["line_indices"], lc["Lin"])


def test_read_twix_uses_imaging_measurement_only(tmp_path) -> None:
    """k-space comes from the LAST measurement; the noise scan is not concatenated."""
    p = tmp_path / "multi.dat"
    noise = np.ones((2, 8), dtype=np.complex64)
    img = np.full((4, 32), 7, dtype=np.complex64)
    _build_vd(
        p,
        [
            {"protocol": _ascconv(seq=r"%AdjSeq%\noise"), "acqs": [noise]},
            {"protocol": _ascconv(seq=r"%SiemensSeq%\trufi"), "acqs": [img]},
        ],
    )
    out = twix.read_twix(p)
    assert out["kspace"].shape == (1, 4, 32)


def test_read_twix_ragged_acquisitions_raise(tmp_path) -> None:
    """Mixed scan types (noise/phasecorr have different sample counts) must not be silently merged."""
    p = tmp_path / "ragged.dat"
    a0 = np.zeros((4, 64), dtype=np.complex64)
    a1 = np.zeros((4, 32), dtype=np.complex64)  # different n_samp
    _build_vd(p, [{"protocol": _ascconv(), "acqs": [a0, a1]}])
    with pytest.raises(ValueError, match=r"[Rr]agged|varying"):
        twix.read_twix(p)


def test_read_twix_vb_kspace_not_implemented(tmp_path) -> None:
    p = tmp_path / "vb.dat"
    _build_vb(p, _ascconv())
    with pytest.raises(NotImplementedError, match="VB"):
        twix.read_twix(p)
