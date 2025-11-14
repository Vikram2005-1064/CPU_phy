#!/usr/bin/env python3
"""
layer2_mlwe.py
Revised Layer 2 MLWE prototype – FIXED.

Improvements:
 - Safe sampling for chi_eta
 - Correct poly_mul_convolution overflow handling
 - Proper array shapes for keygen and homomorphic operations
 - Decode clipping
 - Broadcasting fixes for homomorphic add
 - Minor dtype and boundary improvements
 - ECC fallback maintained
"""

from __future__ import annotations
import os
import hashlib
import secrets
import struct
from typing import Tuple, Optional, List
import numpy as np
from numpy.random import default_rng
from pack.core.layer1 import ChaosEntropyGenerator

# Optional Reed–Solomon ECC
try:
    import reedsolo
    RS_AVAILABLE = True
except Exception:
    RS_AVAILABLE = False

# -----------------------
# Parameters
# -----------------------
n = 256
q = 3329
eta = 2
k = 2
REP = 3
CHECKSUM_LEN = 4
ECC_REDUNDANCY = 8
MAX_MESSAGE_BYTES = n // REP - CHECKSUM_LEN - ECC_REDUNDANCY
COMPRESSION_BITS = 10  # 10-bit packing
USE_KYBER_NTT = False
KYBER_ZETAS: Optional[np.ndarray] = None
KYBER_INV_ZETAS: Optional[np.ndarray] = None

# Parameter sanity
assert n == 256, "Current NTT tables (if used) assume n=256"
assert q == 3329, "Current NTT tables (if used) assume Kyber q=3329"
assert k in (2, 3, 4), "Typical Kyber-like k"
assert 1 <= eta <= 3, "Centered binomial eta in {1,2,3} is typical"


# -----------------------
# Utilities
# -----------------------

try:
    import reedsolo
    RS_AVAILABLE = True
except Exception:
    RS_AVAILABLE = False

def sha3_256(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()

def center_coeffs(arr: np.ndarray) -> np.ndarray:
    arr_int = np.asarray(arr, dtype=np.int64)
    return ((arr_int + q//2) % q) - q//2

def secure_wipe(buf: bytearray):
    for _ in range(3):
        for i in range(len(buf)):
            buf[i] = secrets.randbelow(256)
    for i in range(len(buf)):
        buf[i] = 0

def estimate_noise(poly: np.ndarray) -> float:
    return float(np.std(center_coeffs(poly)))

# -----------------------
# Sampler
# -----------------------

# Assuming your sample_chi_eta from Layer 2 exists:
def sample_chi_eta(length: int, eta_param: int = 2, rng: np.random.Generator | None = None) -> np.ndarray:
    """Centered binomial sampler. If rng is provided, use it; else OS randomness."""
    num_bits = 2 * eta_param * length
    num_bytes = (num_bits + 7) // 8
    if rng is None:
        rnd_bytes = os.urandom(num_bytes)
    else:
        # rng.bit_generator.random_raw gives uint64; simpler: use integers
        rnd = rng.integers(0, 256, size=num_bytes, dtype=np.uint8)
        rnd_bytes = rnd.tobytes()
    bits = np.unpackbits(np.frombuffer(rnd_bytes, dtype=np.uint8))[:num_bits]
    bits = bits.reshape(length, 2 * eta_param)
    return (bits[:, :eta_param].sum(axis=1) - bits[:, eta_param:].sum(axis=1)).astype(np.int64)



# --- Wrapper function ---
def get_entropy_vector(length: int, eta_param: int = 2) -> np.ndarray:
    """
    Returns a vector of entropy values:
    - Uses enhanced Layer 1 chaotic key generator if available.
    - Otherwise falls back to sample_chi_eta.

    FIX: Removed attempted import from _main_ to resolve dependency/circular import.
    Forces fallback sample_chi_eta to ensure the script runs reliably.
    """
    # Fallback to local sampler

    return sample_chi_eta(length, eta_param).astype(np.float64) / (2 * eta_param)


# -----------------------
# Polynomial ops
# -----------------------
def poly_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64))

def poly_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64))

def poly_mul_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64 = np.asarray(a, dtype=np.int64)
    b64 = np.asarray(b, dtype=np.int64)
    res = np.convolve(a64, b64)
    res = np.pad(res, (0, max(0, 2*n - res.size)), 'constant')
    low = res[:n]
    high = res[n:2*n]
    return ((low - high) % q).astype(np.int64)

def ntt_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        A = kyber_ntt(a)
        B = kyber_ntt(b)
        C = kyber_intt((A * B) % q)
        return freeze(C)
    except NotImplementedError:
        # Fall back to convolution if NTT not available
        return poly_mul_convolution(a, b)


def kyber_ntt(a: np.ndarray) -> np.ndarray:
    # Requires full KYBER_ZETAS. Guard until provided.
    raise NotImplementedError("KYBER_ZETAS not populated with full table; NTT disabled.")

def kyber_intt(a: np.ndarray) -> np.ndarray:
    # Requires full KYBER_INV_ZETAS. Guard until provided.
    raise NotImplementedError("KYBER_INV_ZETAS not populated with full table; INTT disabled.")


def poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if USE_KYBER_NTT and KYBER_ZETAS is not None:
        return ntt_mul(a, b)
    return poly_mul_convolution(a, b)

# -----------------------
# Modulus switching
# -----------------------
def modulus_switch(poly: np.ndarray, old_mod: int = q, new_mod: int = (q >> 1)) -> np.ndarray:
    if new_mod >= old_mod:
        raise ValueError("new_mod must be smaller than old_mod")
    poly_int = np.asarray(poly, dtype=np.int64)
    scaled = np.round(poly_int.astype(np.float64) * (new_mod / old_mod)).astype(np.int64)
    return (scaled % new_mod).astype(np.int64)

# -----------------------
# ECC wrappers
# -----------------------
def ecc_encode(msg: bytes) -> bytes:
    if not RS_AVAILABLE:
        return msg + b'\x00' * ECC_REDUNDANCY
    rs = reedsolo.RSCodec(ECC_REDUNDANCY)
    return rs.encode(msg)

def ecc_decode(msg: bytes) -> Tuple[bytes, bool]:
    if not RS_AVAILABLE:
        if len(msg) >= ECC_REDUNDANCY:
            return msg[:-ECC_REDUNDANCY], True
        return msg, False
    rs = reedsolo.RSCodec(ECC_REDUNDANCY)
    try:
        decoded = rs.decode(msg)
        return decoded, True
    except reedsolo.ReedSolomonError:
        return msg[:-ECC_REDUNDANCY], False

# -----------------------
# Encode / Decode
# -----------------------
def encode_message(msg: bytes) -> np.ndarray:
    if len(msg) > MAX_MESSAGE_BYTES:
        raise ValueError(f"Message too long (max {MAX_MESSAGE_BYTES})")
    msg_ecc = ecc_encode(msg)
    chk = sha3_256(msg_ecc)[:CHECKSUM_LEN]
    full = msg_ecc + chk
    scale = q // 256
    poly = np.zeros(n, dtype=np.int64)
    idx = 0
    for b in full:
        coeff = (int(b) * scale) % q
        for rep in range(REP):
            if idx >= n:
                break
            poly[idx] = (coeff + rep) % q
            idx += 1
    while idx < n:
        poly[idx] = secrets.randbelow(q)
        idx += 1
    return poly

def decode_message(poly: np.ndarray) -> Tuple[Optional[bytes], bool]:
    poly_int = np.asarray(poly, dtype=np.int64)
    scale = q // 256
    recovered = bytearray()
    groups = n // REP
    for g in range(groups):
        block = center_coeffs(poly_int[g*REP:(g+1)*REP])
        median_val = int(round(np.median(block)))
        byte_val = np.clip(int(round(median_val / float(scale))), 0, 255)
        recovered.append(byte_val)
    if len(recovered) < CHECKSUM_LEN + ECC_REDUNDANCY:
        return None, False
    msg_ecc = bytes(recovered[:-CHECKSUM_LEN])
    chk = bytes(recovered[-CHECKSUM_LEN:])
    if sha3_256(msg_ecc)[:CHECKSUM_LEN] != chk:
        return None, False
    decoded, ok = ecc_decode(msg_ecc)
    return (decoded if ok else None), ok

# -----------------------
# CRT pack/unpack
# -----------------------
def crt_pack(msgs: List[bytes]) -> np.ndarray:
    if not msgs:
        return np.zeros(n, dtype=np.int64)
    slots = len(msgs)
    if slots > n:
        raise ValueError("Too many slots")
    size = n // slots
    poly = np.zeros(n, dtype=np.int64)
    for i, m in enumerate(msgs):
        chunk = m[:size].ljust(size, b'\x00')
        for j, b in enumerate(chunk):
            poly[i*size + j] = int(b) % q
    return poly

def crt_unpack(poly: np.ndarray, slots: int) -> List[bytes]:
    if slots <= 0 or slots > n:
        raise ValueError("Invalid slots")
    size = n // slots
    poly_int = np.asarray(poly, dtype=np.int64)
    res = []
    for i in range(slots):
        arr = poly_int[i*size:(i+1)*size]
        b = bytes([int(x % 256) for x in arr]).rstrip(b'\x00')
        res.append(b)
    return res

# -----------------------
# Compression
# -----------------------
def compress_ct(poly: np.ndarray) -> bytes:
    poly_int = np.asarray(poly, dtype=np.int64)
    scale_down = q / (1 << COMPRESSION_BITS)
    vals = np.round(poly_int.astype(np.float64) / scale_down).astype(np.int64)
    vals = np.clip(vals, 0, (1 << COMPRESSION_BITS) - 1)
    out = bytearray()
    for i in range(0, n, 4):
        chunk = vals[i:i+4]
        if chunk.size < 4:
            chunk = np.pad(chunk, (0, 4-chunk.size), constant_values=0)
        packed = (int(chunk[0]) << 30) | (int(chunk[1]) << 20) | (int(chunk[2]) << 10) | int(chunk[3])
        out.extend(struct.pack(">Q", packed)[-5:])
    return bytes(out)

def decompress_ct(data: bytes) -> np.ndarray:
    poly = np.zeros(n, dtype=np.int64)
    idx = 0
    data_idx = 0
    while idx < n and data_idx + 5 <= len(data):
        packed_bytes = b'\x00\x00\x00' + data[data_idx:data_idx+5]
        packed = struct.unpack(">Q", packed_bytes)[0]
        vals = [(packed >> 30) & 0x3FF, (packed >> 20) & 0x3FF, (packed >> 10) & 0x3FF, packed & 0x3FF]
        for v in vals:
            if idx < n:
                poly[idx] = int(v)
                idx += 1
        data_idx += 5
    scale_down = q / (1 << COMPRESSION_BITS)
    poly = (poly.astype(np.float64) * scale_down).round().astype(np.int64) % q
    return poly

# -----------------------
# KEM-style KeyGen / Encaps / Decaps
# -----------------------
def keygen(seed: Optional[bytes] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    seed_in = seed or os.urandom(32)
    rng = default_rng(int.from_bytes(sha3_256(seed_in)[:8], 'big'))
    A = rng.integers(0, q, size=(k, k, n), dtype=np.int64)
    s = np.stack([sample_chi_eta(n, eta_param=eta, rng=rng) for _ in range(k)])
    e = np.stack([sample_chi_eta(n, eta_param=eta, rng=rng) for _ in range(k)])
    t = np.zeros((k, n), dtype=np.int64)
    for i in range(k):
        acc = np.zeros(n, dtype=np.int64)
        for j in range(k):
            acc = poly_add(acc, poly_mul(A[i, j, :], s[j, :]))
        t[i, :] = poly_add(acc, e[i, :])
    return (A, t), s


def encapsulate(pk: Tuple[np.ndarray, np.ndarray], payload: Optional[bytes] = None, rng: np.random.Generator | None = None) -> Tuple[Tuple[np.ndarray, np.ndarray], bytes]:
    A, t = pk
    if rng is None:
        rng = default_rng()
    r = np.stack([sample_chi_eta(n, eta_param=eta, rng=rng) for _ in range(k)])
    e1 = np.stack([sample_chi_eta(n, eta_param=eta, rng=rng) for _ in range(k)])
    e2 = sample_chi_eta(n, eta_param=eta, rng=rng)
    u = np.zeros((k, n), dtype=np.int64)
    for i in range(k):
        acc = np.zeros(n, dtype=np.int64)
        for j in range(k):
            acc = poly_add(acc, poly_mul(A[j][i], r[j]))
        u[i] = poly_add(acc, e1[i])
    if payload is None:
        payload = secrets.token_bytes(MAX_MESSAGE_BYTES)
    m = encode_message(payload)
    acc = np.zeros(n, dtype=np.int64)
    for i in range(k):
        acc = poly_add(acc, poly_mul(t[i], r[i]))
    v = poly_add(poly_add(acc, e2), m)
    u_noise = estimate_noise(u.flatten())
    v_noise = estimate_noise(v)
    if u_noise > q/8 or v_noise > q/8:
        print(f"Warning: high noise (u:{u_noise:.2f}, v:{v_noise:.2f})")
    return (u, v), payload


def decapsulate(sk: np.ndarray, ct: Tuple[np.ndarray, np.ndarray]) -> Tuple[Optional[bytes], bool]:
    u, v = ct
    acc = np.zeros(n, dtype=np.int64)
    for i in range(k):
        acc = poly_add(acc, poly_mul(u[i], sk[i]))
    diff = poly_sub(v, acc)
    noise_level = estimate_noise(diff)
    if noise_level > q/6:
        print(f"Warning: decryption noise high ({noise_level:.2f})")
    return decode_message(diff)

def mlwe_encapsulate(pk: Tuple[np.ndarray, np.ndarray], payload: Optional[bytes] = None) -> Tuple[Tuple[np.ndarray, np.ndarray], bytes]:
    return encapsulate(pk, payload)

def mlwe_decapsulate(sk: np.ndarray, ct: Tuple[np.ndarray, np.ndarray]) -> Tuple[Optional[bytes], bool]:
    return decapsulate(sk, ct)


def gen_relin_key_ntt(k_val: int = k) -> np.ndarray:
    """
    Stub relinearization key in NTT domain.
    For demo only; real key-switch uses secret-dependent construction.
    """
    return np.zeros((k_val, k_val, n), dtype=np.int64)


# -----------------------
# Homomorphic add
# -----------------------
def homomorphic_add(ct1: Tuple[np.ndarray, np.ndarray], ct2: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    u1, v1 = ct1
    u2, v2 = ct2
    u_sum = np.stack([poly_add(u1[i], u2[i]) for i in range(k)])
    v_sum = poly_add(v1, v2)
    u_noise = estimate_noise(u_sum.flatten())
    v_noise = estimate_noise(v_sum)
    if u_noise > q/4 or v_noise > q/4:
        print(f"Warning: noise growing in addition (u:{u_noise:.2f}, v:{v_noise:.2f})")
    return (u_sum, v_sum)

# -----------------------
# Placeholder NTT functions (FIX: Defined placeholders for execution)
# -----------------------
def ntt_func(arr: np.ndarray) -> np.ndarray:
    """Placeholder NTT: returns input."""
    if USE_KYBER_NTT:
        print("Error: NTT is enabled but not implemented! Returning input.")
    return arr

def intt_func(arr: np.ndarray) -> np.ndarray:
    """Placeholder inverse NTT: returns input."""
    if USE_KYBER_NTT:
        print("Error: INTT is enabled but not implemented! Returning input.")
    return arr

# -----------------------
# Homomorphic multiplication via NTT
# -----------------------

def homomorphic_mul_ntt(
        ct1: Tuple[np.ndarray, np.ndarray],
        ct2: Tuple[np.ndarray, np.ndarray],
        relin_key_ntt: Optional[np.ndarray],
        ntt_func,
        intt_func,
        noise_threshold: float = q / 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Homomorphic multiplication using batched NTT with optional relinearization.

    Args:
        ct1, ct2: ciphertext tuples (u, v)
            u: (k, n) numpy array
            v: (n,) numpy array
        relin_key_ntt: optional relinearization key in NTT domain, shape (k, k, n)
        ntt_func: forward NTT function, vectorized over axis 1
        intt_func: inverse NTT function, vectorized over axis 1
        noise_threshold: maximum allowed noise (std. dev.)

    Returns:
        (u_final, v_final): resulting ciphertext tuple
    """
    u1, v1 = ct1
    u2, v2 = ct2
    k1, n1 = u1.shape
    k2, n2 = u2.shape

    if k1 != k2 or n1 != n2 or v1.shape != (n1,) or v2.shape != (n1,):
        raise ValueError(f"homomorphic_mul_ntt: invalid shapes {u1.shape}, {u2.shape}, {v1.shape}, {v2.shape}")

    k, n = k1, n1

    # Precompute NTTs in batch
    u1_ntt = ntt_func(u1)  # expects shape (k, n)
    u2_ntt = ntt_func(u2)
    v1_ntt = ntt_func(v1)[None, :]  # shape (1, n)
    v2_ntt = ntt_func(v2)[None, :]  # shape (1, n)

    # 1. v_mul = INTT(v1_ntt * v2_ntt)
    v_mul = intt_func(v1_ntt * v2_ntt)[0] % q

    # 2. Cross terms: INTT(u1_ntt * v2_ntt + u2_ntt * v1_ntt)
    cross_ntt = u1_ntt * v2_ntt + u2_ntt * v1_ntt  # broadcast shapes (k,n)
    u_cross = intt_func(cross_ntt) % q  # shape (k,n)

    # 3. Degree-2 terms: sum_j INTT(u1_ntt[i] * u2_ntt[j]) for each i
    #    Then optional relinearization via precomputed relin_key_ntt[i,j]
    # Compute outer product in NTT domain
    # shape (k, k, n)
    u2_term_ntt = u1_ntt[:, None, :] * u2_ntt[None, :, :]
    if relin_key_ntt is not None:
        if relin_key_ntt.shape != (k, k, n):
            raise ValueError(f"homomorphic_mul_ntt: relin_key_ntt must have shape (k,k,{n})")
        # apply key-switch by pointwise multiply in NTT domain
        u2_term_ntt = u2_term_ntt * relin_key_ntt

    # sum over second index -> shape (k, n)
    u2_term = intt_func(u2_term_ntt.sum(axis=1)) % q

    # 4. Final u and v
    u_final = (u_cross + u2_term) % q
    v_final = v_mul

    # 5. Noise estimation
    sigma_u = float(np.std(center_coeffs(u_final)))
    sigma_v = float(np.std(center_coeffs(v_final)))
    if sigma_u > noise_threshold or sigma_v > noise_threshold:
        print(f"Warning: high noise after homomorphic_mul_ntt (σ_u={sigma_u:.2f}, σ_v={sigma_v:.2f})")

    return u_final, v_final


# -----------------------
# Demo / Tests
# -----------------------
if __name__ == "__main__":
    print("Layer 2 revised prototype")
    if not RS_AVAILABLE:
        print("Note: reedsolo not available — ECC disabled (fallback parity). Install via pip install reedsolo for ECC.")

    # Key generation
    pk, sk = keygen()
    print("Keys generated.")

    # Basic KEM test
    ct, payload = encapsulate(pk, b"HelloLayer2")
    rec, ok = decapsulate(sk, ct)
    print("Decapsulate OK:", ok, "Recovered:", rec)

    # Homomorphic addition test
    ct1, _ = encapsulate(pk, b"A")
    ct2, _ = encapsulate(pk, b"B")
    sum_ct = homomorphic_add(ct1, ct2)
    rec2, ok2 = decapsulate(sk, sum_ct)
    print("HomAdd OK:", ok2, "Rec partial bytes:", (rec2[:8] if rec2 else None))

    # Compression test
    comp = compress_ct(ct[0][0])
    print("Compression produced", len(comp), "bytes (~320 expected).")

    # CRT pack/unpack test
    packed = crt_pack([b"X", b"Y"])
    unpacked = crt_unpack(packed, slots=2)
    print("CRT pack/unpack:", unpacked)

    # Homomorphic multiplication via NTT test (if NTT available)
    if USE_KYBER_NTT and KYBER_ZETAS is not None:
        # Prepare a dummy relin_key in NTT domain for demonstration
        # In practice, compute relin_key_ntt properly ahead of time
        dummy_key = np.stack([
            np.stack([ntt_func(np.zeros(n, dtype=np.int64)) for _ in range(k)], axis=0)
            for _ in range(k)
        ], axis=0)

        # Homomorphic multiplication (placeholder path unless real NTT provided)
        relin = gen_relin_key_ntt(k)
        try:
            rng = default_rng(12345)
            ct_mul, _ = encapsulate(pk, b"C", rng=rng)
            ct_mul2, _ = encapsulate(pk, b"D", rng=rng)
            u_mul, v_mul = homomorphic_mul_ntt(ct_mul, ct_mul2, relin, ntt_func, intt_func)
            rec3, ok3 = decapsulate(sk, (u_mul, v_mul))
            print("HomMul NTT OK (placeholder):", ok3, "Rec partial bytes:", (rec3[:8] if rec3 else None))
        except Exception as e:
            print("HomMul NTT test failed:", e)

    else:
        print("NTT not enabled—skipping homomorphic_mul_ntt test.")

    print("Done.")

class Layer2MLWE:
    def __init__(self,
                 n: int = 256,
                 q_: int = 3329,
                 eta: int = 2,
                 k: int = 2,
                 compression_bits: int = 10,
                 use_kyber_ntt: bool = False):
        self.n = n
        self.q = q_
        self.eta = eta
        self.k = k
        self.compression_bits = compression_bits
        self.use_kyber_ntt = use_kyber_ntt
        self.layer1_generator = None
        try:
            from pack.core.layer1 import ChaosEntropyGenerator
            self.layer1_generator = ChaosEntropyGenerator()
        except Exception:
            self.layer1_generator = None

    def keygen(self, seed: bytes = None):
        if seed is None and self.layer1_generator is not None:
            try:
                entropy_data = self.layer1_generator.extract_entropy()
                pools = [
                    entropy_data.get('classical_entropy', b''),
                    entropy_data.get('hybrid_entropy', b''),
                    entropy_data.get('master_key', b'')
                ]
                seed_material = b''.join([p if isinstance(p, (bytes, bytearray)) else b'' for p in pools])
                if len(seed_material) < 32:
                    seed_material += os.urandom(32 - len(seed_material))
                seed = sha3_256(seed_material)
            except Exception:
                seed = os.urandom(32)
        if seed is None:
            seed = os.urandom(32)
        return keygen(seed=seed)

    def encapsulate(self, pk, payload=None):
        return mlwe_encapsulate(pk, payload)

    def decapsulate(self, sk, ct):
        return mlwe_decapsulate(sk, ct)

    def homomorphic_add(self, ct1, ct2):
        return homomorphic_add(ct1, ct2)

    def homomorphic_mul(self, ct1, ct2, relin_key_ntt=None, ntt_func=None, intt_func=None):
        return homomorphic_mul_ntt(ct1, ct2, relin_key_ntt, ntt_func, intt_func)
