"""A small pure-Python GGUF reader: the header and the key/value metadata, never the tensor data.

Enough to know a model's architecture, context length, layer and head counts (for the KV-cache estimate), quantisation and whether it ships
a chat template, and to give a file a digest that notices when the model changed.
"""

from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path
from typing import Any, BinaryIO, Optional

from .errors import GaltonError

MAGIC = b"GGUF"
DIGEST_BYTES = 16 * 1024 * 1024
#: Arrays longer than this are counted but not kept (the tokenizer vocabulary has 150k entries).
KEEP_ARRAY_UP_TO = 64
#: Numeric arrays are kept up to this length: per-layer values (head counts of a hybrid model, one per layer) must survive.
KEEP_NUMERIC_UP_TO = 4096

_FIXED = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
T_STRING, T_ARRAY = 8, 9

FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
              15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
              25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE"}
_QUANT_IN_NAME = re.compile(r"(?<![A-Za-z0-9])((?:IQ|Q)\d(?:_[A-Z0-9]+)*|BF16|F16|F32|MXFP4)(?![A-Za-z0-9])", re.I)


class GgufError(GaltonError):
    def __init__(self, message: str, hint: str = ""):
        super().__init__("invalid", message, hint or "Check that the file is a complete GGUF (not a partial download).")


def _read(fh: BinaryIO, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise GgufError("The file ends in the middle of the GGUF header.")
    return data


def _string(fh: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read(fh, 8))
    if length > 1 << 24:
        raise GgufError(f"Implausible string length {length} in the GGUF header.")
    return _read(fh, length).decode("utf-8", "replace")


def _value(fh: BinaryIO, kind: int) -> Any:
    if kind in _FIXED:
        fmt, size = _FIXED[kind]
        return struct.unpack(fmt, _read(fh, size))[0]
    if kind == T_STRING:
        return _string(fh)
    if kind == T_ARRAY:
        (elem,) = struct.unpack("<I", _read(fh, 4))
        (count,) = struct.unpack("<Q", _read(fh, 8))
        if elem in _FIXED:
            size = _FIXED[elem][1]
            if count <= KEEP_NUMERIC_UP_TO:
                return list(struct.unpack(f"<{count}{_FIXED[elem][0][1:]}", _read(fh, size * count)))
            fh.seek(size * count, 1)
            return {"array": count}
        if elem == T_STRING:
            keep = count <= KEEP_ARRAY_UP_TO
            items = []
            for _ in range(count):
                if keep:
                    items.append(_string(fh))
                else:
                    (n,) = struct.unpack("<Q", _read(fh, 8))
                    fh.seek(n, 1)
            return items if keep else {"array": count}
        if elem == T_ARRAY:  # the format allows arrays of arrays: read them all, keep them when short
            nested = [_value(fh, T_ARRAY) for _ in range(count)]
            return nested if count <= KEEP_ARRAY_UP_TO else {"array": count}
        raise GgufError(f"Unsupported array element type {elem}.")
    raise GgufError(f"Unknown GGUF value type {kind}.")


def read_kv(path: str | Path) -> dict[str, Any]:
    """Every key/value pair of the header (long arrays are replaced by ``{"array": length}``)."""
    path = Path(path)
    try:
        fh = path.open("rb")
    except OSError as exc:
        raise GaltonError("not_found", "file_unreadable", path=path, detail=exc.strerror or str(exc)) from exc
    with fh:
        if _read(fh, 4) != MAGIC:
            raise GgufError(f"{path.name} is not a GGUF file (bad magic).")
        (version,) = struct.unpack("<I", _read(fh, 4))
        if version not in (2, 3):
            raise GgufError(f"GGUF version {version} is not supported (only 2 and 3).")
        tensor_count, kv_count = struct.unpack("<QQ", _read(fh, 16))
        if kv_count > 100_000:
            raise GgufError(f"Implausible metadata count {kv_count}.")
        kv: dict[str, Any] = {"_version": version, "_tensor_count": tensor_count}
        for _ in range(kv_count):
            key = _string(fh)
            (kind,) = struct.unpack("<I", _read(fh, 4))
            kv[key] = _value(fh, kind)
        return kv


def _num(value: Any) -> Optional[float]:
    """A scalar from a metadata value; a per-layer array becomes its largest entry."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list) and value:
        nums = [v for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
        return float(max(nums)) if nums else None
    return None


def quant_from_name(name: str) -> str:
    m = _QUANT_IN_NAME.search(name)
    return m.group(1).upper() if m else ""


def params_from_text(text: str) -> Optional[float]:
    """Billions of parameters from ``27B`` / ``7b`` / ``600M`` in a name (``None`` when absent)."""
    m = re.search(r"(?<![A-Za-z0-9])(\d+(?:[.,]\d+)?)\s*([BbMm])(?![A-Za-z])", text.replace("_", " ").replace("-", " "))
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    return round(value if m.group(2).lower() == "b" else value / 1000, 3)


def summarize(kv: dict[str, Any], filename: str = "") -> dict[str, Any]:
    """The fields Galton uses, with ``None`` for what the file does not say."""
    arch = str(kv.get("general.architecture") or "")

    def key(suffix: str) -> Optional[float]:
        return _num(kv.get(f"{arch}.{suffix}"))

    heads, kv_heads = key("attention.head_count"), key("attention.head_count_kv")
    # hybrid models (linear attention or state-space layers between attention layers) list the KV heads per layer, 0 where a layer
    # keeps no KV cache: only the others count for the cache estimate
    per_layer = kv.get(f"{arch}.attention.head_count_kv")
    kv_layers = sum(1 for v in per_layer if isinstance(v, (int, float)) and v > 0) if isinstance(per_layer, list) else None
    embed = key("embedding_length")
    head_dim = key("attention.key_length") or ((embed / heads) if embed and heads else None)
    ftype = kv.get("general.file_type")
    quant = FILE_TYPES.get(int(ftype), "") if isinstance(ftype, int) else ""
    size_label = str(kv.get("general.size_label") or "")
    vocab = kv.get("tokenizer.ggml.tokens")
    out = {
        "architecture": arch, "name": str(kv.get("general.name") or ""), "context_length": int(key("context_length") or 0) or None,
        "block_count": int(key("block_count") or 0) or None, "kv_layers": kv_layers or None, "embedding_length": int(embed) if embed else None,
        "head_count": int(heads) if heads else None, "head_count_kv": int(kv_heads or heads) if (kv_heads or heads) else None,
        "head_dim": int(head_dim) if head_dim else None, "expert_count": int(key("expert_count") or 0) or None,
        "quant": quant or quant_from_name(filename), "file_type": int(ftype) if isinstance(ftype, int) else None,
        "has_chat_template": bool(kv.get("tokenizer.chat_template")), "size_label": size_label,
        "params_b": params_from_text(size_label) or params_from_text(filename), "vocab_size": vocab["array"] if isinstance(vocab, dict) else None,
        "is_projector": arch == "clip" or str(kv.get("general.type") or "") == "mmproj", "tensor_count": kv.get("_tensor_count"),
    }
    return out


def read_meta(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    meta = summarize(read_kv(path), path.name)
    meta["size_bytes"] = path.stat().st_size
    return meta


def file_digest(path: str | Path) -> str:
    """``sha256`` of the first 16 MB plus the size: cheap on a 20 GB file and changes when the model does."""
    path = Path(path)
    h = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as fh:
        h.update(fh.read(DIGEST_BYTES))
    h.update(str(size).encode())
    return f"sha256-16m:{h.hexdigest()[:40]}"


# ------------------------------------------------------------------------------------------------- memory estimate
HEADROOM_MB = 600
FILE_FACTOR = 1.08


def kv_cache_mb(meta: dict[str, Any], context: int) -> float:
    """f16 KV cache for ``context`` tokens. Exact when the metadata has the layer and head counts, else 0.15 GB per 1k tokens for a 20B+ model, scaled down."""
    layers, kv_heads, head_dim = meta.get("kv_layers") or meta.get("block_count"), meta.get("head_count_kv"), meta.get("head_dim")
    if layers and kv_heads and head_dim:
        return 2 * layers * context * kv_heads * head_dim * 2 / (1024 * 1024)
    params = meta.get("params_b") or 27.0
    return 0.15 * (context / 1000) * min(1.0, params / 20.0) * 1024


def estimate_vram_mb(size_bytes: int, meta: dict[str, Any], context: int, mmproj_bytes: int = 0) -> int:
    """File size × 1.08 + KV cache + projector + 600 MB of headroom, in MiB."""
    weights = size_bytes * FILE_FACTOR / (1024 * 1024)
    return int(weights + kv_cache_mb(meta, context) + mmproj_bytes * FILE_FACTOR / (1024 * 1024) + HEADROOM_MB)
