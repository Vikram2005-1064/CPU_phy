import numpy as np
from scipy import linalg
import hashlib

class QuantumChaosEngine:
    def __init__(self, dimension: int):
        self.dimension = dimension

    def generate_quantum_chaos_matrix(self, seed_entropy: float) -> np.ndarray:
        d = self.dimension
        theta = np.linspace(0, 2 * np.pi, d, endpoint=False, dtype=np.float64)
        momentum = (np.fft.fftfreq(d) * 2.0 * np.pi).astype(np.float64)
        H_kinetic = np.diag((momentum ** 2) / 2.0)
        kick_strength = 0.5 + float(seed_entropy) * 10.0
        H_potential = -kick_strength * np.cos(np.subtract.outer(theta, theta)).astype(np.float64)
        dt = 0.01
        H = H_kinetic + H_potential
        U = linalg.expm((-1j * dt) * H.astype(np.complex128))
        return U

    def extract_quantum_entropy(self, U_matrix: np.ndarray, iterations: int = 1000) -> bytes:
        d = self.dimension
        rng = np.random.default_rng()
        psi = (rng.random(d) + 1j * rng.random(d)).astype(np.complex128)
        psi /= np.linalg.norm(psi)
        hasher = hashlib.sha3_256()
        for _ in range(iterations):
            psi = U_matrix @ psi
            norm = np.linalg.norm(psi)
            if not np.isfinite(norm) or norm == 0.0:
                psi = (rng.random(d) + 1j * rng.random(d)).astype(np.complex128)
                psi /= np.linalg.norm(psi)
            else:
                psi /= norm
            mags = np.abs(psi).astype(np.float64, copy=False)
            phases = np.angle(psi).astype(np.float64, copy=False)
            hasher.update(mags.tobytes())
            hasher.update(phases.tobytes())
        return hasher.digest()
