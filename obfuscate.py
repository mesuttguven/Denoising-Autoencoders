"""
Byte-level obfuscations.

The first six are reproduced EXACTLY as coded in the original notebook
(Using_DAE_for_Detecting_Obfuscated_Malware.ipynb, cell 17), not as
described in the manuscript.  Differences between the two matter:

  Encryption               XOR 0xFF on every byte (bit complement)
  Control_Flow_Obfuscation XOR 0xFF on every 4th byte, from offset 0
  Data_Obfuscation         +1 mod 256 on every byte
  String_Obfuscation       XOR 0xFF on one byte every 32, from offset 512
  Code_Packing             zlib.compress of the whole file
  File_Header_Modification keeps bytes [0:128] and OVERWRITES THE REST
                           WITH 0xFF.  This destroys the file body, so it
                           is NOT invertible.  It is kept as a
                           "destructive control": no method can recover
                           the content, so any gain over identity must
                           come from something other than reconstruction.

`invertible=True` means apply/invert round-trip exactly (checked on every
file at build time).  For those techniques the "oracle" condition equals
the clean condition by construction.

UPX is an optional real-world packer, enabled only if `upx` is on PATH.
"""
import os
import shutil
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def _arr(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8).copy()


def code_packing(b):
    return zlib.compress(b)


def code_packing_inv(b):
    return zlib.decompress(b)


def encryption(b):
    return (_arr(b) ^ 0xFF).tobytes()


def header_mod(b):
    return bytes(b[:128]) + b"\xff" * max(len(b) - 128, 0)


def control_flow(b):
    a = _arr(b)
    a[0::4] ^= 0xFF
    return a.tobytes()


def data_obf(b):
    return (_arr(b) + np.uint8(1)).tobytes()          # uint8 wraps mod 256


def data_obf_inv(b):
    return (_arr(b) - np.uint8(1)).tobytes()


def string_obf(b):
    a = _arr(b)
    a[512::32] ^= 0xFF
    return a.tobytes()


def upx_pack(b):
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, "in.bin"), os.path.join(d, "out.bin")
        with open(src, "wb") as f:
            f.write(b)
        r = subprocess.run(["upx", "-q", "-9", "-o", dst, src],
                           capture_output=True, timeout=120)
        if r.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError("upx failed")
        with open(dst, "rb") as f:
            return f.read()


@dataclass(frozen=True)
class Technique:
    name: str
    apply: Callable[[bytes], bytes]
    invert: Optional[Callable[[bytes], bytes]]   # None => not invertible

    @property
    def invertible(self) -> bool:
        return self.invert is not None


_ALL = [
    Technique("Code_Packing", code_packing, code_packing_inv),
    Technique("Control_Flow_Obfuscation", control_flow, control_flow),   # XOR is an involution
    Technique("Data_Obfuscation", data_obf, data_obf_inv),
    Technique("Encryption", encryption, encryption),
    Technique("File_Header_Modification", header_mod, None),
    Technique("String_Obfuscation", string_obf, string_obf),
]
if shutil.which("upx"):
    _ALL.append(Technique("UPX", upx_pack, None))

TECHNIQUES = {t.name: t for t in _ALL}
