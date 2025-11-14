import hashlib
from typing import Dict, Any, Optional
import numpy as np
from pack.core.layer1 import ChaosEntropyGenerator

def generate_master_key(*args, **kwargs):
    generator = ChaosEntropyGenerator(*args, **kwargs)
    entropy_data = generator.extract_entropy()
    return {
        "master_key": generator.derive_master_key(entropy_data),
        "entropy_data": entropy_data
    }


def extract_entropy_only(num_balls: int = 300,
                         sim_frames: int = 2000,
                         **kwargs) -> Dict[str, Any]:
    """
    Extract only entropy data without key derivation.

    Args:
        num_balls: Number of balls in simulation
        sim_frames: Number of simulation frames
        **kwargs: additional parameters passed to ChaosEntropyGenerator

    Returns:
        Entropy data as a dictionary
    """
    generator = ChaosEntropyGenerator(num_balls=num_balls,
                                      sim_frames=sim_frames,
                                      **kwargs)
    return generator.extract_entropy()


def derive_key_from_entropy(classical_entropy: bytes,
                            quantum_entropy: bytes,
                            pbkdf2_iterations: int = 200000) -> str:
    """
    Derive a master key from existing entropy data.

    Args:
        classical_entropy: Classical entropy bytes
        quantum_entropy: Quantum entropy bytes
        pbkdf2_iterations: PBKDF2 iteration count

    Returns:
        Master key as hex string
    """
    from pack.core.kdf import QuantumResistantKDF
    kdf = QuantumResistantKDF(pbkdf2_iters=pbkdf2_iterations)
    master_key = kdf.derive_master_key(classical_entropy, quantum_entropy, quantum_entropy)
    return master_key.hex()


def get_layer1_info() -> Dict[str, Any]:
    """
    Return static information about Layer 1 capabilities.
    """
    return {
        'layer': 1,
        'name': 'Quantum-Resistant Chaos Entropy Generator',
        'description': 'Generates entropy and quantum-resistant keys using chaotic physical simulations',
        'entropy_sources': ['classical_chaos', 'quantum_chaos', 'hardware_rng', 'lattice_mixing'],
        'key_derivation': ['Argon2id', 'PBKDF2_SHA3_512_fallback'],
        'security_level': '256-512 bits depending on configuration'
    }
# --------------- Modular helpers ---------------
def montgomery_reduce(a: np.ndarray | int) -> np.ndarray | int:
    # Kyber uses Montgomery reduction; here a simple mod q is acceptable but keep vectorized.
    if isinstance(a, np.ndarray):
        return (a % q).astype(np.int64)
    return int(a % q)

def barrett_reduce(a: np.ndarray | int) -> np.ndarray | int:
    # Barrett reduction tuned for q=3329. For simplicity use % q; placeholder for speed.
    return montgomery_reduce(a)

def freeze(a: np.ndarray) -> np.ndarray:
    # Map to [0,q)
    return (np.asarray(a, dtype=np.int64) % q).astype(np.int64)
