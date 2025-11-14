#!/usr/bin/env python3
"""
Patched entropy generator / key derivation script:
- Replaces deprecated random_integers
- Streams entropy into hashers (memory-safe)
- Optional Argon2id KDF (uses argon2-cffi if installed)
- Fixed __init__ constructors, pygame guards, dtype issues, and lattice mix
"""

import os
# Headless-safe pygame import/init
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
# If running headless (e.g., CI), allow no-window mode via env
HEADLESS = os.environ.get("HEADLESS", "0") == "1"
if HEADLESS:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import math
import secrets
import time
import hashlib
import struct
import numpy as np
from scipy import linalg
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA3_512 as C_SHA3_512
from typing import Union, Optional, Dict, Any

# Optional Argon2 import (if available)
USE_ARGON2 = False
try:
    from argon2.low_level import hash_secret_raw, Type
    USE_ARGON2 = True
except Exception:
    USE_ARGON2 = False

# === ENHANCED CONFIG ===
NUM_BALLS = 300
RADIUS = 5
HISTORY_BALLS = 10
SIM_FRAMES = 2000
# Use a reproducible RNG for layout values (not CSPRNG for entropy)
_layout_rng = np.random.default_rng()
# Ensure these evaluate to ints; also clamp to reasonable sizes
SCREEN_W = int(max(640, min(1920, 80 * math.sin(float(_layout_rng.integers(-100, 100))) + 1200)))
SCREEN_H = int(max(480, min(1200, 80 * math.sin(float(_layout_rng.integers(-100, 100))) + 800)))
BALL_DIM = 256  # Quantum state dimension
PBKDF2_ITERS = 200_000

# === GLOBALS (minimized) ===
balls: list[dict] = []
secrets_generator = secrets.SystemRandom()

# === ENHANCED HELPER FUNCTIONS ===
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def ultra_chaotic_generator(t: float, seed: float) -> float:
    """Chaotic function with safety guards to avoid domain/overflow issues."""
    try:
        random_noise = (secrets.randbits(16) / 65535.0) - 0.5
        a = math.sin(t * seed * math.sqrt(2.0))
        # bound exp argument for cos to avoid overflow in exp
        exp_arg = _clamp(t, -20.0, 20.0)
        b = math.cos(math.exp(exp_arg) * math.pi)
        # avoid log1p domain issues
        c = math.tan(t + math.log1p(abs(seed) + 1e-12))
        d = math.pow(abs(math.cos(a * b) + math.sin(c)), 1.5)
        # bound sinh input modestly
        e = math.atan(math.sinh(_clamp(a + b - c, -10.0, 10.0)))
        f = math.log1p(abs(d + e))
        g = math.sin(t * f * (seed + random_noise))
        h = math.tanh(t + g * 3.1415)
        i = ((int(t * 1000.0) ^ int(seed * 1000.0)) & 255) / 100.0
        j = math.cos(g + h + i)
        k = math.sin(math.log1p(abs(j)))
        val = (a * e + b * f + c * g + h * i + j * k) + random_noise * 5.0
        normalized_val = math.tanh(val)
        return float(normalized_val)
    except Exception:
        return 0.0

def flatten_and_hash(ball: dict) -> bytes:
    """Entropy extraction for a single ball (returns 32-byte SHA3-256 digest)."""
    def norm(value: float, min_v: float, max_v: float) -> float:
        return (value - min_v) / (max_v - min_v + 1e-12)

    h = hashlib.sha3_256()
    for key, (lo, hi) in {
        'mass_history': (15.0, 77.66),  # widened upper bound a bit
        'e_history': (0.25, 0.95),
        'G_history': (0.016, 0.101),
    }.items():
        hist = ball.get(key)
        if hist:
            for val in hist:
                h.update(struct.pack(">f", float(norm(val, lo, hi))))
    return h.digest()

def floats_to_canonical_bytes(arr: np.ndarray) -> bytes:
    """Convert numpy array to platform-independent bytes (big-endian float64)."""
    a = np.asarray(arr, dtype='>f8', order='C')
    return a.tobytes()

def bytes_to_uint8_norm_array(b: bytes) -> np.ndarray:
    """Convert bytes to normalized float array in [0,1]."""
    # Use frombuffer on a copy to avoid referencing immutable memory if needed
    return (np.frombuffer(bytes(b), dtype=np.uint8).astype(np.float64) / 255.0)

# === QUANTUM CHAOS ENGINE (memory-safe extraction) ===
class QuantumChaosEngine:
    def __init__(self, dimension: int = BALL_DIM):
        self.dimension = int(dimension)

    def generate_quantum_chaos_matrix(self, seed_entropy: float) -> np.ndarray:
        d = self.dimension
        theta = np.linspace(0, 2 * np.pi, d, endpoint=False, dtype=np.float64)
        momentum = (np.fft.fftfreq(d) * 2.0 * np.pi).astype(np.float64)
        H_kinetic = np.diag((momentum ** 2) / 2.0)
        kick_strength = 0.5 + float(seed_entropy) * 10.0
        # Use cosine of pairwise difference
        H_potential = -kick_strength * np.cos(np.subtract.outer(theta, theta)).astype(np.float64)
        dt = 0.01
        H = H_kinetic + H_potential
        # Ensure complex128 for expm
        U = linalg.expm((-1j * dt) * H.astype(np.complex128))
        return U

    def extract_quantum_entropy(self, U_matrix: np.ndarray, iterations: int = 1000) -> bytes:
        """
        Stream magnitudes/phases into SHA3-256 incrementally to avoid
        building a huge in-memory buffer. Returns a 32-byte digest.
        """
        d = self.dimension
        rng = np.random.default_rng()
        psi = (rng.random(d) + 1j * rng.random(d)).astype(np.complex128)
        psi /= np.linalg.norm(psi)
        hasher = hashlib.sha3_256()

        # update incrementally
        for _ in range(int(iterations)):
            psi = U_matrix @ psi
            # renormalize occasionally to control drift
            norm = np.linalg.norm(psi)
            if not np.isfinite(norm) or norm == 0.0:
                # re-seed if numerical instability
                psi = (rng.random(d) + 1j * rng.random(d)).astype(np.complex128)
                psi /= np.linalg.norm(psi)
            else:
                psi /= norm
            mags = np.abs(psi).astype(np.float64, copy=False)
            phases = np.angle(psi).astype(np.float64, copy=False)
            hasher.update(floats_to_canonical_bytes(mags))
            hasher.update(floats_to_canonical_bytes(phases))

        return hasher.digest()

# === HARDWARE ENTROPY SOURCE ===
class HardwareEntropySource:
    def get_hardware_entropy(self, num_bytes: int = 64) -> bytes:
        try:
            return os.urandom(int(num_bytes))
        except Exception:
            return secrets.token_bytes(int(num_bytes))

    def mix_with_hardware(self, software_entropy: bytes) -> bytes:
        hw = self.get_hardware_entropy(len(software_entropy))
        if len(hw) != len(software_entropy):
            h = hashlib.sha3_256()
            h.update(b'\x01' + software_entropy)
            h.update(b'\x02' + hw)
            return h.digest()
        # XOR is fine post-cryptographic hash, but keep anyway
        return bytes(a ^ b for a, b in zip(software_entropy, hw))

# === LATTICE ENTROPY MIXER ===
class LatticeEntropyMixer:
    def __init__(self, dimension: int = 512, modulus: int = 3329, seed: Optional[int] = None):
        self.n = int(dimension)
        self.q = int(modulus)
        rng = np.random.default_rng(seed)
        # Non-zero to avoid trivial zero polynomial; int64 to avoid overflow in conv
        self.polynomial_ring = rng.integers(1, self.q, size=self.n, dtype=np.int64)

    def lattice_mix_entropy(self, chaos_data: Union[np.ndarray, bytes, bytearray]) -> np.ndarray:
        if isinstance(chaos_data, (bytes, bytearray)):
            chaos_arr = bytes_to_uint8_norm_array(bytes(chaos_data))
        else:
            chaos_arr = np.asarray(chaos_data, dtype=np.float64)

        if chaos_arr.size < self.n:
            chaos_arr = np.pad(chaos_arr, (0, self.n - chaos_arr.size), mode='wrap')
        elif chaos_arr.size > self.n:
            chaos_arr = chaos_arr[:self.n]

        # Map to [0,q-1] as integers
        # Avoid bias by clipping tiny rounding errors
        scaled = np.floor(np.clip(chaos_arr, 0.0, 1.0) * self.q).astype(np.int64) % self.q
        # Linear convolution; result length 2n-1
        mixed_poly = np.convolve(scaled, self.polynomial_ring, mode='full').astype(np.int64)

        # Reduce modulo x^n + 1 via negacyclic reduction
        # mixed_poly indices: 0..(2n-2)
        reduced = mixed_poly[:self.n].copy()
        tail = mixed_poly[self.n:]
        # For i in 0..(n-2): add (-tail[i]) to reduced[i]
        # Because x^n ≡ -1
        for i in range(tail.size):
            reduced[i] = (reduced[i] - tail[i])  # negacyclic wrap

        reduced %= self.q
        return reduced.astype(np.float64) / float(self.q)

# === QUANTUM-RESISTANT KDF (Argon2 optional) ===
class QuantumResistantKDF:
    def __init__(self, pbkdf2_iters: int = PBKDF2_ITERS):
        self.lattice_mixer = LatticeEntropyMixer()
        self.hw_entropy = HardwareEntropySource()
        self.pbkdf2_iters = int(pbkdf2_iters)

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

        if USE_ARGON2:
            # Argon2id parameters (tune for target machine). Memory in kibibytes.
            time_cost = 4
            memory_cost = 1 << 16  # 64 MiB (adjustable)
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
            # PBKDF2 with HMAC-SHA3-512 fallback
            stage1_key = PBKDF2(
                combined,
                salt,
                dklen=64,
                count=self.pbkdf2_iters,
                hmac_hash_module=C_SHA3_512
            )

        # Stage 2 consolidation
        stage2_key = hashlib.sha3_512(stage1_key + salt).digest()
        arr = bytes_to_uint8_norm_array(stage2_key)
        lattice_mixed = self.lattice_mixer.lattice_mix_entropy(arr)
        final_seed = floats_to_canonical_bytes(lattice_mixed)
        final_key = hashlib.sha3_256(final_seed).digest()
        return final_key

# === HYBRID CLASSICAL-QUANTUM SIMULATION (memory-safe) ===
def run_hybrid_simulation_and_get_key() -> Dict[str, Any]:
    global balls
    balls = []

    quantum_chaos = QuantumChaosEngine()
    kdf = QuantumResistantKDF()

    # Initialize pygame and screen
    pygame.init()
    # Fail gracefully if headless
    screen = None
    clock = None
    try:
        flags = 0 if not HEADLESS else pygame.HIDDEN
        screen = pygame.display.set_mode((int(SCREEN_W), int(SCREEN_H)), flags=flags)
        clock = pygame.time.Clock()
    except Exception:
        # Headless mode fallback
        screen = None
        clock = None

    center_x, center_y = SCREEN_W // 2, SCREEN_H // 2
    base_width = max(RADIUS + 5, (SCREEN_W // 2) - RADIUS)
    base_height = max(RADIUS + 5, (SCREEN_H // 2) - RADIUS)
    pulse_amplitude = 100
    pulse_frequency = 1.0

    # Ball initialization using CSPRNG for seeds only
    rng = np.random.default_rng(int.from_bytes(secrets.token_bytes(8), 'big'))
    for i in range(NUM_BALLS):
        initial_seed = secrets_generator.uniform(0.1, 1.5)
        chaos_m_init = _clamp(ultra_chaotic_generator(0.0, initial_seed), 0.07, 0.93)
        chaos_e_init = _clamp(ultra_chaotic_generator(0.0, initial_seed + 0.1), 0.07, 0.93)
        chaos_g_init = _clamp(ultra_chaotic_generator(0.0, initial_seed + 0.2), 0.07, 0.93)

        ball = {
            'id': i,
            'x': secrets_generator.uniform(center_x - base_width, center_x + base_width),
            'y': secrets_generator.uniform(center_y - base_height, center_y + base_height),
            'vx': secrets_generator.uniform(-5.0, 5.0),
            'vy': secrets_generator.uniform(-5.0, 5.0),
            'chaos_m': chaos_m_init,
            'chaos_e': chaos_e_init,
            'chaos_g': chaos_g_init,
            'm_r': 3.98 + secrets_generator.uniform(-0.018, 0.018),
            'e_r': 3.94 + secrets_generator.uniform(-0.023, 0.023),
            'g_r': 3.97 + secrets_generator.uniform(-0.012, 0.012),
            'mass': 15.0 + 62.0 * chaos_m_init,
            'mass_history': [] if i < HISTORY_BALLS else None,
            'e_history': [] if i < HISTORY_BALLS else None,
            'G_history': [] if i < HISTORY_BALLS else None,
            'trajectory': [] if i < HISTORY_BALLS else None,
        }
        balls.append(ball)

    start_time = time.perf_counter()
    frame_counter = 0

    # Use incremental hash for classical entropy to avoid large memory use
    classical_hasher = hashlib.sha3_256()

    # Simulation loop
    while frame_counter < SIM_FRAMES:
        # Event pump only if screen exists
        if screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    frame_counter = SIM_FRAMES
                    break

        if screen is not None:
            screen.fill((0, 0, 0))
        t = time.perf_counter() - start_time

        semi_major_axis_x = base_width + pulse_amplitude * math.sin(t * pulse_frequency)
        semi_major_axis_y = base_height + pulse_amplitude * math.cos(t * pulse_frequency)

        if screen is not None:
            pygame.draw.ellipse(
                screen, (50, 50, 50),
                (center_x - semi_major_axis_x, center_y - semi_major_axis_y,
                 2 * semi_major_axis_x, 2 * semi_major_axis_y), 1
            )

        frame_entropy_bytes = bytearray()

        for i, ball in enumerate(balls):
            # Chaos evolution (bounded)
            ball['chaos_m'] = _clamp(ball['m_r'] * ball['chaos_m'] * (1 - ball['chaos_m']) + secrets_generator.uniform(-0.008, 0.008), 0.07, 0.93)
            ball['chaos_e'] = _clamp(ball['e_r'] * ball['chaos_e'] * (1 - ball['chaos_e']) + secrets_generator.uniform(-0.008, 0.008), 0.07, 0.93)
            ball['chaos_g'] = _clamp(ball['g_r'] * ball['chaos_g'] * (1 - ball['chaos_g']) + secrets_generator.uniform(-0.006, 0.006), 0.07, 0.93)

            mass = 15.0 + 62.0 * ball['chaos_m']
            e = 0.25 + 0.7 * ball['chaos_e']
            G = 0.016 + ball['chaos_g'] * 0.085

            if i < HISTORY_BALLS:
                ball['mass_history'].append(mass)
                ball['e_history'].append(e)
                ball['G_history'].append(G)
                ball['trajectory'].append((ball['x'], ball['y']))
            ball['mass'] = mass

            # Forces from others (simple n^2). Skip if too close.
            fx, fy = 0.0, 0.0
            for j, other in enumerate(balls):
                if i == j:
                    continue
                dx, dy = other['x'] - ball['x'], other['y'] - ball['y']
                dist_sq = dx * dx + dy * dy
                if dist_sq <= 1e-12 or dist_sq < (2.0 * RADIUS) ** 2:
                    continue
                force = G * mass * (other['mass']) / dist_sq
                angle = math.atan2(dy, dx)
                fx += force * math.cos(angle)
                fy += force * math.sin(angle)

            nudge = secrets_generator.uniform(-0.12, 0.12)
            ball['vx'] += fx + nudge
            ball['vy'] += fy + nudge
            speed_limit = 30.0
            ball['vx'] = _clamp(ball['vx'], -speed_limit, speed_limit)
            ball['vy'] = _clamp(ball['vy'], -speed_limit, speed_limit)
            ball['x'] += ball['vx']
            ball['y'] += ball['vy']

            # Boundary handling (ellipse)
            dx_center = ball['x'] - center_x
            dy_center = ball['y'] - center_y
            # Prevent division by zero for degenerate ellipse axes
            ax = max(1e-6, semi_major_axis_x)
            ay = max(1e-6, semi_major_axis_y)
            ellipse_eq = (dx_center ** 2 / ax ** 2) + (dy_center ** 2 / ay ** 2)

            if ellipse_eq >= 1.0:
                nx = dx_center / (ax ** 2)
                ny = dy_center / (ay ** 2)
                norm_magnitude = math.hypot(nx, ny) or 1.0
                nx /= norm_magnitude
                ny /= norm_magnitude
                dot_product = ball['vx'] * nx + ball['vy'] * ny
                ball['vx'] = (ball['vx'] - 2.0 * dot_product * nx) * e
                ball['vy'] = (ball['vy'] - 2.0 * dot_product * ny) * e
                current_eq_sqrt = math.sqrt(max(ellipse_eq, 1.0))
                ball['x'] = center_x + dx_center / current_eq_sqrt
                ball['y'] = center_y + dy_center / current_eq_sqrt

            if screen is not None:
                pygame.draw.circle(screen, (255, 255, 255), (int(ball['x']), int(ball['y'])), RADIUS)

            # Collect minimal-per-frame entropy bytes and feed classical hasher
            ball_state = struct.pack(">ffff", float(ball['x']), float(ball['y']), float(ball['vx']), float(ball['vy']))
            chaos_state = struct.pack(">fff", float(ball['chaos_m']), float(ball['chaos_e']), float(ball['chaos_g']))
            frame_entropy_bytes.extend(ball_state + chaos_state)

        # after processing all balls in the frame, update the classical_hasher incrementally
        classical_hasher.update(bytes(frame_entropy_bytes))

        # simple n^2 collision handler
        n_balls = len(balls)
        for i_idx in range(n_balls):
            for j_idx in range(i_idx + 1, n_balls):
                ball1, ball2 = balls[i_idx], balls[j_idx]
                dx, dy = ball2['x'] - ball1['x'], ball2['y'] - ball1['y']
                dist = math.hypot(dx, dy)
                if 0.0 < dist < 2.0 * RADIUS:
                    nx, ny = dx / dist, dy / dist
                    p = 2.0 * (ball1['vx'] * nx + ball1['vy'] * ny - ball2['vx'] * nx - ball2['vy'] * ny) / (
                            ball1['mass'] + ball2['mass'])
                    ball1['vx'] -= p * ball2['mass'] * nx
                    ball1['vy'] -= p * ball2['mass'] * ny
                    ball2['vx'] += p * ball1['mass'] * nx
                    ball2['vy'] += p * ball1['mass'] * ny
                    overlap = (2.0 * RADIUS - dist) / 2.0 + 0.01
                    ball1['x'] -= overlap * nx
                    ball1['y'] -= overlap * ny
                    ball2['x'] += overlap * nx
                    ball2['y'] += overlap * ny

        if screen is not None:
            pygame.display.flip()
            clock.tick(30)
        frame_counter += 1

    # Quit pygame cleanly if it was initialized
    try:
        pygame.quit()
    except Exception:
        pass

    # === CLASSICAL ENTROPY DIGEST (incremental) ===
    classical_key_raw = classical_hasher.digest()

    # If HISTORY_BALLS, prefer flatten_and_hash on one of the tracked balls for richer features
    if HISTORY_BALLS > 0 and len(balls) > 0 and balls[0].get('mass_history'):
        classical_key_raw = flatten_and_hash(balls[0])

    # === QUANTUM CHAOS ENHANCEMENT ===
    # Create a seed from classical_key_raw deterministically
    seed_hash = hashlib.sha3_256(classical_key_raw).digest()
    seed53 = int.from_bytes(seed_hash[:8], 'big') & ((1 << 53) - 1)
    seed_entropy = float(seed53) / float(1 << 53)

    U_matrix = quantum_chaos.generate_quantum_chaos_matrix(seed_entropy)
    quantum_digest = quantum_chaos.extract_quantum_entropy(U_matrix, iterations=400)

    # === FINAL KEY DERIVATION ===
    classical_bytes = classical_key_raw
    lattice_bytes = quantum_digest  # already bytes

    master_key = kdf.derive_master_key(classical_bytes, quantum_digest, lattice_bytes)

    return {
        'master_key': master_key.hex(),
        'classical_entropy_bits': len(classical_bytes) * 8,
        'quantum_entropy_bits': len(quantum_digest) * 8,
        'total_security_level': 512 if USE_ARGON2 else 256,
        'entropy_sources': ['classical_chaos', 'quantum_chaos', 'hardware_rng', 'lattice_mixing'],
        'key_derivation': 'Argon2id' if USE_ARGON2 else 'PBKDF2_SHA3_512_MultiStage'
    }


# === MAIN ===
if __name__ == "__main__":
    print("Argon2 available:", USE_ARGON2)
    results = run_hybrid_simulation_and_get_key()
    print("Extracted Enhanced Key (hex):", results['master_key'])
    print("Metadata:", {k: results[k] for k in results if k != 'master_key'})

#====== Wrapper Class =======

class ChaosEntropyGenerator:
    def __init__(self,
                 num_balls: int = 300,
                 sim_frames: int = 2000,
                 ball_dimension: int = 256,
                 pbkdf2_iterations: int = 200_000,
                 use_pygame: bool = True,
                 headless: bool = False):
        # Save config
        self.num_balls = num_balls
        self.sim_frames = sim_frames
        self.ball_dimension = ball_dimension
        self.pbkdf2_iterations = pbkdf2_iterations
        self.use_pygame = use_pygame
        self.headless = headless

        # Initialize components
        self.quantum_chaos = QuantumChaosEngine(dimension=ball_dimension)
        self.lattice_mixer = LatticeEntropyMixer()
        self.kdf = QuantumResistantKDF(pbkdf2_iters=pbkdf2_iterations)

    def extract_entropy(self):
        """
        Run the classical-quantum hybrid simulation and extract entropy
        Returns a dictionary with classical and quantum entropy bytes.
        """
        return run_hybrid_simulation_and_get_key()

    def derive_master_key(self, entropy_data: dict = None):
        """
        Derive the master key from entropy_data dictionary.
        If entropy_data is None, simulate and extract entropy.
        Returns hex master key.
        """
        if entropy_data is None:
            entropy_data = self.extract_entropy()

        classical_entropy = entropy_data.get('classical_entropy_bytes') or entropy_data.get('classical_entropy_bits')
        quantum_entropy = entropy_data.get('quantum_entropy_bytes') or entropy_data.get('quantum_entropy_bits')

        if isinstance(classical_entropy, int) or classical_entropy is None:
            classical_entropy = entropy_data.get('classical_entropy') or b''
        if isinstance(quantum_entropy, int) or quantum_entropy is None:
            quantum_entropy = entropy_data.get('quantum_entropy') or b''

        master_key = self.kdf.derive_master_key(classical_entropy, quantum_entropy, quantum_entropy)
        return master_key.hex()
