"""
Pack - Multi-Layer Cryptographic System

A comprehensive cryptographic system combining:
- Layer 1: Quantum-Resistant Chaos Key Derivation  
- Layer 2: MLWE Homomorphic Encryption

Usage:
    # Layer 1 - Entropy and Key Generation
    from pack import ChaosEntropyGenerator, generate_master_key

    # Layer 2 - Homomorphic Encryption
    from pack import Layer2MLWE, keygen, encrypt

    # Combined usage
    layer1 = ChaosEntropyGenerator()
    entropy_data = layer1.extract_entropy()
    master_key = layer1.derive_master_key(entropy_data)

    layer2 = Layer2MLWE()
    pk, sk = layer2.keygen(seed=master_key[:32])
"""

# Layer 1
from .core.layer1 import ChaosEntropyGenerator

# Layer 2
from .core.layer2 import (
    Layer2MLWE,
    keygen as mlwe_keygen,
    encapsulate as mlwe_encapsulate,
    decapsulate as mlwe_decapsulate,
    homomorphic_add,
    homomorphic_mul_ntt
)

# Utils
from .utils.helpers import generate_master_key, extract_entropy_only

# Version info
__version__ = "1.0.0"
__author__ = "VikRAM_64GB"

# Public API
__all__ = [
    # Layer 1
    "ChaosEntropyGenerator",

    # Layer 2
    "Layer2MLWE",
    "mlwe_keygen",
    "mlwe_encapsulate",
    "mlwe_decapsulate",
    "homomorphic_add",
    "homomorphic_mul_ntt",
    
    #Layer 3
    "Layer3NeuralPRNG",

    # Utils
    "generate_master_key",
    "extract_entropy_only"
]
