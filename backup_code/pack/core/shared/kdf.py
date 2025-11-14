import hashlib
import secrets
import struct
import time
import os
from typing import Optional

from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA3_512 as C_SHA3_512
try:
    from argon2.low_level import hash_secret_raw, Type
    USE_ARGON2 = True
except ImportError:
    USE_ARGON2 = False

from .lattice_mixer import LatticeEntropyMixer
from .hardware_entropy import HardwareEntropySource
from typing import Union

class QuantumResistantKDF:
    def __init__(self, pbkdf2_iters: int = 200000):
        self.lattice_mixer = LatticeEntropyMixer()
        self.hw_entropy = HardwareEntropySource()
        self.pbkdf2_iters = pbkdf2_iters
        self.USE_ARGON2 = USE_ARGON2

    @staticmethod
    def _combine_entropy_sources(classical_ent: bytes, quantum_ent: bytes, lattice_ent: bytes, hw_ent: bytes) -> bytes:
        h = hashlib.sha3_256()
        h.update(b'\x01' + classical_ent)
        h.update(b'\x02' + quantum_ent)
        h.update(b'\x03' + lattice_ent)
        h.update(b'\x04' + hw_ent)
        return h.digest()

    def _generate_quantum_salt(self) -> bytes:
        hw_random = self.hw_entropy.get_hardware_entropy(32)
        time_entropy = struct.pack(">Q", time.time_ns() & ((1 << 64) - 1))
        pid = os.getpid()
        pgrp = os.getpgrp() if hasattr(os, "getpgrp") else 0
        proc_entropy = struct.pack(">II", pid & 0xFFFFFFFF, pgrp & 0xFFFFFFFF)
        extra = secrets.token_bytes(16)
        salt_material = hw_random + time_entropy + proc_entropy + extra
        return hashlib.sha3_512(salt_material).digest()[:32]

    def derive_master_key(self, classical_entropy: bytes, quantum_entropy: bytes, lattice_entropy: bytes) -> bytes:
        hw_entropy = self.hw_entropy.get_hardware_entropy(32)
        combined = self._combine_entropy_sources(classical_entropy, quantum_entropy, lattice_entropy, hw_entropy)
        salt = self._generate_quantum_salt()

        if self.USE_ARGON2:
            time_cost = 4
            memory_cost = 1 << 16  # 64 MiB
            parallelism = 4
            outlen = 64
            stage1_key = hash_secret_raw(
                secret=combined,
                salt=salt,
                time_cost=time_cost,
                memory_cost=memory_cost,
                parallelism=parallelism,
                hash_len=outlen,
                type=Type.ID
            )
        else:
            # PBKDF2 fallback
            stage1_key = PBKDF2(
                combined,
                salt,
                dklen=64,
                count=self.pbkdf2_iters,
                hmac_hash_module=C_SHA3_512
            )

        stage2_key = hashlib.sha3_512(stage1_key + salt).digest()
        arr = np.frombuffer(stage2_key, dtype=np.uint8).astype(np.float64) / 255.0
        lattice_mixed = self.lattice_mixer.lattice_mix_entropy(arr)
        final_seed = lattice_mixed.tobytes()
        final_key = hashlib.sha3_256(final_seed).digest()
        return final_key
