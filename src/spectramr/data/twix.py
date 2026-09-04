"""Pure-Python Siemens TWIX ``.dat`` raw-acquisition decoder.

TWIX is Siemens' proprietary raw-data container (the output of ``twixtools`` /
``mapVBVD`` territory). There is no public schema, so this is a *format decoder*
written from the documented on-disk layout, with **no third-party dependency** —
matching the repo's pure-stdlib ``profile_bart`` / h5py-only ``IsmrmrdStrategy``
precedent. It is the single source of truth for TWIX parsing, consumed by:

* ``spectramr.data.io_strategies.TwixStrategy`` — full k-space loading;
* ``spectramr.data.profiler.profile_twix`` — header-only physics profiling.

The module is intentionally **torch-free** (numpy + stdlib only) so the profiler,
which never materialises voxels, can import it freely.

Two on-disk variants exist:

* **VD/VE** (modern, multi-RAID): an ``id``/``n_scans`` prelude, a table of
  per-measurement ``offset``/``length`` entries, then for each measurement a header
  buffer section followed by ``sScanHeader``/``sChannelHeader``-prefixed k-space.
  The *imaging* measurement is the last entry (earlier ones are noise/adjust).
* **VB** (older, single measurement): a header-length prelude, the header buffers,
  then MDH-prefixed k-space.

Header parsing (``parse_twix_header``) supports both. k-space reading (``read_twix``)
implements VD/VE — it returns the imaging measurement's *flat* acquisition list
``[n_acq, n_coil, n_samp]`` (like ``IsmrmrdStrategy``) and **raises** rather than
guess a gridded ``[Lin, Par, …]`` volume from the loop counters (the dataset layer
owns that mapping) or decode an unimplemented VB payload (pitfall #9: no silent
fallback).
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

__all__ = [
    "TwixHeader",
    "TwixMeasurement",
    "detect_twix_version",
    "parse_ascconv",
    "parse_twix_header",
    "read_twix",
]

# --- VD on-disk constants -----------------------------------------------------------
_ENTRY_STRUCT = struct.Struct("<IIQQ64s64s")  # measId, fileId, offset, length, pat, prot
_ENTRY_SIZE = _ENTRY_STRUCT.size  # 152
_SCAN_HEADER_SIZE = 192  # VD sScanHeader
_CHANNEL_HEADER_SIZE = 32  # VD sChannelHeader
# byte offsets within sScanHeader that this decoder reads
_OFF_EVALMASK = 40  # aulEvalInfoMask[0]
_OFF_SAMPLES = 48  # ushSamplesInScan
_OFF_CHANNELS = 50  # ushUsedChannels
_OFF_LIN = 52  # sLoopCounter.Lin
# The VD sLoopCounter is 14 consecutive uint16 starting at _OFF_LIN. The reader
# historically surfaced only Lin; the others (Sli = slice, Set = the bSSFP
# phase-cycle, Ave = average/acquisition, Eco = echo) are what a phase-cycled
# multi-slice / dual-echo acquisition needs to be gridded — without them
# n_acq = n_pe is underdetermined. Names per the Siemens ICE mdh.
_LOOP_NAMES = (
    "Lin",
    "Ave",
    "Sli",
    "Par",
    "Eco",
    "Phs",
    "Rep",
    "Set",
    "Seg",
    "Ida",
    "Idb",
    "Idc",
    "Idd",
    "Ide",
)
_LOOP_STRUCT = struct.Struct("<14H")  # read as a block at _OFF_LIN
_ACQEND = 0x1  # aulEvalInfoMask bit 0 — end of a measurement

# VD detection thresholds (the twixtools/mapVBVD heuristic): a VD file opens with a
# small ``id`` and an ``n_scans`` count in ``[1, 64]``; a VB file opens with a (large)
# header-length word, and its second word is the start of ASCII protocol text (a
# big integer when read as u32), so it can never masquerade as ``n_scans <= 64``.
_VD_ID_MAX = 10000
_VD_NSCANS_MAX = 64


@dataclass(frozen=True)
class TwixMeasurement:
    """One measurement entry in a VD multi-RAID file."""

    meas_id: int
    file_id: int
    offset: int
    length: int
    patient_name: str
    protocol_name: str


@dataclass(frozen=True)
class TwixHeader:
    """Header-only physics metadata parsed from a TWIX protocol (no k-space read)."""

    version: str  # "vb" | "vd"
    n_measurements: int
    field_strength_t: float | None
    sequence: str | None  # basename of tSequenceFileName, lowercased (e.g. "trufi")
    n_coils: int | None
    matrix: list[int] | None  # [base_resolution, phase_encoding_lines]
    n_contrasts: int | None  # lContrasts (echoes)
    te_ms: list[float] | None
    tr_ms: list[float] | None
    flip_deg: list[float] | None
    vendor: str = "Siemens"  # TWIX is a Siemens-only format


# --- ASCCONV protocol parsing -------------------------------------------------------

_ASCCONV_BEGIN = re.compile(r"### ASCCONV BEGIN[^\n]*###")
_ASCCONV_END = "### ASCCONV END ###"
_INDEXED_KEY = re.compile(r"^(?P<base>[A-Za-z_.]+)\[(?P<idx>\d+)\]$")
_COILS_RE = re.compile(r'iMaxNoOfRxChannels"\s*>\s*\{\s*(\d+)')


def parse_ascconv(text: str) -> dict[str, str]:
    """Parse the flat ``key = value`` ASCCONV (MeasYaps) block into a dict.

    The block is delimited by ``### ASCCONV BEGIN … ###`` / ``### ASCCONV END ###``;
    each interior line is ``key\\t = \\tvalue`` (whitespace varies). Returns ``{}`` when
    no block is present (the caller decides whether that is fatal). Surrounding double
    quotes on values are stripped.
    """
    begin = _ASCCONV_BEGIN.search(text)
    if begin is None:
        return {}
    body = text[begin.end() :]
    end = body.find(_ASCCONV_END)
    if end != -1:
        body = body[:end]

    out: dict[str, str] = {}
    for line in body.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"')
        if key:
            out[key] = value
    return out


def _indexed_floats(ascconv: dict[str, str], base: str, scale: float = 1.0) -> list[float] | None:
    """Collect an indexed numeric array (``base[0]``, ``base[1]``, …), sorted, scaled."""
    items: list[tuple[int, float]] = []
    for key, value in ascconv.items():
        m = _INDEXED_KEY.match(key)
        if m and m.group("base") == base:
            try:
                items.append((int(m.group("idx")), float(value) * scale))
            except ValueError:
                continue
    if not items:
        return None
    items.sort()
    return [v for _, v in items]


def _int_or_none(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _sequence_name(ascconv: dict[str, str]) -> str | None:
    """Basename of ``tSequenceFileName`` (``%SiemensSeq%\\trufi`` -> ``trufi``), lowered."""
    raw = ascconv.get("tSequenceFileName")
    if not raw:
        return None
    tail = re.split(r"[\\/]", raw.strip())[-1]
    return tail.lower() or None


# --- low-level file structure -------------------------------------------------------


def detect_twix_version(fh: BinaryIO) -> tuple[str, int]:
    """Sniff the TWIX variant from the first 8 bytes. Returns ``(version, n_scans)``.

    ``n_scans`` is the VD measurement count (1 for VB). Raises ``ValueError`` on a file
    too short to be TWIX.
    """
    fh.seek(0)
    head = fh.read(8)
    if len(head) < 8:
        raise ValueError("file is too short to be a TWIX .dat (< 8 bytes).")
    first, second = struct.unpack("<II", head)
    if first < _VD_ID_MAX and 0 < second <= _VD_NSCANS_MAX:
        return "vd", second
    return "vb", 1


def _read_measurement_table(fh: BinaryIO, n_scans: int) -> list[TwixMeasurement]:
    """Read the VD per-measurement entry table (one 152-byte entry per scan)."""
    fh.seek(8)
    out: list[TwixMeasurement] = []
    for _ in range(n_scans):
        raw = fh.read(_ENTRY_SIZE)
        if len(raw) < _ENTRY_SIZE:
            break
        meas_id, file_id, offset, length, pat, prot = _ENTRY_STRUCT.unpack(raw)
        out.append(
            TwixMeasurement(
                meas_id=meas_id,
                file_id=file_id,
                offset=offset,
                length=length,
                patient_name=pat.split(b"\x00", 1)[0].decode("latin-1"),
                protocol_name=prot.split(b"\x00", 1)[0].decode("latin-1"),
            )
        )
    return out


def _read_protocol_text(fh: BinaryIO, version: str, meas_offset: int) -> tuple[str, int]:
    """Read a measurement's header-buffer text. Returns ``(text, scans_start_offset)``.

    For both VB and VD the header section begins with a ``uint32`` length covering the
    whole section (including the length word); the protocol buffers follow and the
    k-space scans begin immediately after. We decode the whole section as latin-1 — the
    interspersed binary buffer-length prefixes are harmless because every field we read
    is delimited by ASCII markers (``### ASCCONV …``, ``iMaxNoOfRxChannels``).
    """
    base = meas_offset if version == "vd" else 0
    fh.seek(base)
    raw_len = fh.read(4)
    if len(raw_len) < 4:
        raise ValueError("TWIX header section is truncated (no length word).")
    (hdr_len,) = struct.unpack("<I", raw_len)
    payload = fh.read(max(hdr_len - 4, 0))
    return payload.decode("latin-1", "replace"), base + hdr_len


def _read_used_channels(fh: BinaryIO, scans_start: int) -> int | None:
    """Peek the first VD scan header for ``ushUsedChannels`` (coil count) — no payload read."""
    fh.seek(scans_start)
    head = fh.read(_SCAN_HEADER_SIZE)
    if len(head) < _SCAN_HEADER_SIZE:
        return None
    (evalmask,) = struct.unpack_from("<I", head, _OFF_EVALMASK)
    if evalmask & _ACQEND:
        return None
    (n_chan,) = struct.unpack_from("<H", head, _OFF_CHANNELS)
    return int(n_chan) or None


def _header_from_protocol(
    version: str,
    n_measurements: int,
    protocol_text: str,
    n_coils: int | None,
) -> TwixHeader:
    """Build a :class:`TwixHeader` from a measurement's protocol text."""
    if "ASCCONV" not in protocol_text:
        raise ValueError(
            "not a TWIX .dat: no ASCCONV protocol block found in the header section "
            "(refusing to guess — pitfall #9)."
        )
    asc = parse_ascconv(protocol_text)

    base = _int_or_none(asc.get("sKSpace.lBaseResolution"))
    pe = _int_or_none(asc.get("sKSpace.lPhaseEncodingLines"))
    matrix = [base, pe] if base is not None and pe is not None else None

    if n_coils is None:
        m = _COILS_RE.search(protocol_text)
        n_coils = int(m.group(1)) if m else None

    return TwixHeader(
        version=version,
        n_measurements=n_measurements,
        field_strength_t=_float_or_none(asc.get("sProtConsistencyInfo.flNominalB0")),
        sequence=_sequence_name(asc),
        n_coils=n_coils,
        matrix=matrix,
        n_contrasts=_int_or_none(asc.get("lContrasts")),
        te_ms=_indexed_floats(asc, "alTE", scale=1e-3),  # microseconds -> ms
        tr_ms=_indexed_floats(asc, "alTR", scale=1e-3),
        flip_deg=_indexed_floats(asc, "adFlipAngleDegree"),
    )


def parse_twix_header(path: str | Path) -> TwixHeader:
    """Parse the imaging measurement's protocol header (header-only, no k-space read).

    For VD this is the *last* measurement (earlier entries are noise/adjust scans). The
    coil count is read from the first scan header's ``ushUsedChannels`` when available,
    falling back to the XProtocol ``iMaxNoOfRxChannels`` token. Raises ``ValueError`` if
    the file is not TWIX (too short, or no ASCCONV block).
    """
    p = Path(path)
    with open(p, "rb") as fh:
        version, n_scans = detect_twix_version(fh)
        if version == "vd":
            table = _read_measurement_table(fh, n_scans)
            if not table:
                raise ValueError(f"TWIX {p.name}: VD measurement table is empty.")
            meas = table[-1]  # imaging measurement
            protocol_text, scans_start = _read_protocol_text(fh, version, meas.offset)
            n_coils = _read_used_channels(fh, scans_start)
            return _header_from_protocol(version, len(table), protocol_text, n_coils)
        protocol_text, _ = _read_protocol_text(fh, version, 0)
        return _header_from_protocol(version, 1, protocol_text, None)


# --- k-space reading (VD/VE) --------------------------------------------------------


def _read_vd_acquisitions(fh: BinaryIO, scans_start: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the imaging measurement's flat acquisition list from ``scans_start``.

    Returns ``(kspace[n_acq, n_coil, n_samp] complex64, loop_counters[n_acq, 14]
    int)`` — the full ``sLoopCounter`` per acquisition (column order ``_LOOP_NAMES``;
    column 0 is ``Lin``). Stops at the ACQEND scan. Raises if the acquisitions are
    ragged (varying samples/channels) — TWIX interleaves noise/phase-correction
    scans with different geometries, and silently padding/truncating/dropping them
    would corrupt the data (pitfall #9). The dataset layer that knows the sequence
    resolves the counter→axis mapping.
    """
    fh.seek(scans_start)
    acqs: list[np.ndarray] = []
    counters: list[tuple[int, ...]] = []
    shapes: set[tuple[int, int]] = set()
    while True:
        head = fh.read(_SCAN_HEADER_SIZE)
        if len(head) < _SCAN_HEADER_SIZE:
            break
        (evalmask,) = struct.unpack_from("<I", head, _OFF_EVALMASK)
        if evalmask & _ACQEND:
            break
        (n_samp,) = struct.unpack_from("<H", head, _OFF_SAMPLES)
        (n_chan,) = struct.unpack_from("<H", head, _OFF_CHANNELS)
        loop = _LOOP_STRUCT.unpack_from(head, _OFF_LIN)  # 14 uint16 (Lin, Ave, Sli, …)
        shapes.add((n_chan, n_samp))

        scan = np.empty((n_chan, n_samp), dtype=np.complex64)
        for c in range(n_chan):
            fh.read(_CHANNEL_HEADER_SIZE)  # per-channel header (unused fields)
            raw = fh.read(n_samp * 8)  # n_samp interleaved complex64
            if len(raw) < n_samp * 8:
                raise ValueError("TWIX k-space truncated mid-acquisition.")
            scan[c] = np.frombuffer(raw, dtype=np.complex64)
        acqs.append(scan)
        counters.append(loop)

    if not acqs:
        raise ValueError("TWIX imaging measurement contains no acquisition scans.")
    if len(shapes) != 1:
        raise ValueError(
            "ragged TWIX acquisitions (varying samples/channels across scans: "
            f"{sorted(shapes)}) — refusing to pad/truncate/drop silently (pitfall #9). "
            "Mixed noise/phase-correction scans need the sequence-specific loop-counter "
            "map, which the dataset layer owns."
        )
    return np.stack(acqs, axis=0), np.asarray(counters, dtype=np.int64)


def read_twix(path: str | Path) -> dict[str, Any]:
    """Decode a TWIX ``.dat`` imaging measurement into a standard IO dict.

    Returns ``{'kspace': complex64 [n_acq, n_coil, n_samp], 'line_indices': int
    [n_acq], 'loop_counters': {name: int[n_acq]}, 'header': TwixHeader,
    'metadata': {...}}``. ``loop_counters`` exposes the full ``sLoopCounter`` per
    acquisition (``Lin``/``Ave``/``Sli``/``Par``/``Eco``/``Phs``/``Rep``/``Set``/…)
    so a multi-slice, phase-cycled (``Set``) or dual-echo (``Eco``) acquisition can
    be binned onto its true axes rather than reshaped positionally. VD/VE only — VB
    k-space decoding is unimplemented and **raises** (no silent fallback).
    """
    p = Path(path)
    with open(p, "rb") as fh:
        version, n_scans = detect_twix_version(fh)
        if version != "vd":
            raise NotImplementedError(
                f"TWIX {p.name}: VB k-space decoding is not implemented (only the "
                "header is parsed for VB). Re-export as VD/VE, or convert with "
                "twixtools. Refusing to decode a format we cannot verify (pitfall #9)."
            )
        table = _read_measurement_table(fh, n_scans)
        if not table:
            raise ValueError(f"TWIX {p.name}: VD measurement table is empty.")
        meas = table[-1]
        protocol_text, scans_start = _read_protocol_text(fh, version, meas.offset)
        n_coils = _read_used_channels(fh, scans_start)
        header = _header_from_protocol(version, len(table), protocol_text, n_coils)
        kspace, loop_counters = _read_vd_acquisitions(fh, scans_start)

    loop = {name: loop_counters[:, i] for i, name in enumerate(_LOOP_NAMES)}
    return {
        "kspace": kspace,
        "line_indices": loop["Lin"],  # back-compat alias for sLoopCounter.Lin
        "loop_counters": loop,
        "header": header,
        "metadata": {
            "format": "twix",
            "twix_version": version,
            "n_measurements": len(table),
            "source_path": str(p),
        },
    }
