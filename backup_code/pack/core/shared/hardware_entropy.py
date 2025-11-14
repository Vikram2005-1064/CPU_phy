import os
import secrets
import hashlib

class HardwareEntropySource:
    def get_hardware_entropy(self, num_bytes: int = 64) -> bytes:
        try:
            return os.urandom(num_bytes)
        except Exception:
            return secrets.token_bytes(num_bytes)

    def mix_with_hardware(self, software_entropy: bytes) -> bytes:
        hw = self.get_hardware_entropy(len(software_entropy))
        if len(hw) != len(software_entropy):
            h = hashlib.sha3_256()
            h.update(b'\x01' + software_entropy)
            h.update(b'\x02' + hw)
            return h.digest()
        return bytes(a ^ b for a, b in zip(software_entropy, hw))
