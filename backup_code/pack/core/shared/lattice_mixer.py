import numpy as np
from typing import Union

class LatticeEntropyMixer:
    def __init__(self, dimension: int = 512, modulus: int = 3329, seed: int = None):
        self.n = dimension
        self.q = modulus
        rng = np.random.default_rng(seed)
        self.polynomial_ring = rng.integers(1, self.q, size=self.n, dtype=np.int64)

    def lattice_mix_entropy(self, chaos_data: Union[np.ndarray, bytes, bytearray]) -> np.ndarray:
        if isinstance(chaos_data, (bytes, bytearray)):
            chaos_arr = np.frombuffer(bytes(chaos_data), dtype=np.uint8).astype(np.float64) / 255.0
        else:
            chaos_arr = np.asarray(chaos_data, dtype=np.float64)

        if chaos_arr.size < self.n:
            chaos_arr = np.pad(chaos_arr, (0, self.n - chaos_arr.size), mode='wrap')
        elif chaos_arr.size > self.n:
            chaos_arr = chaos_arr[:self.n]

        scaled = np.floor(np.clip(chaos_arr, 0.0, 1.0) * self.q).astype(np.int64) % self.q
        mixed_poly = np.convolve(scaled, self.polynomial_ring, mode='full').astype(np.int64)

        reduced = mixed_poly[:self.n].copy()
        tail = mixed_poly[self.n:]
        for i in range(tail.size):
            reduced[i] = (reduced[i] - tail[i])
        reduced %= self.q
        return reduced.astype(np.float64) / float(self.q)
