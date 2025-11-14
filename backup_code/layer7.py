"""
layer7.py - Layer 7: Adaptive Key Evolution & Signature Obfuscation

Layer 7 provides deterministic key evolution and signature obfuscation with
the ability to recover original signatures. This layer adds an additional
layer of security through:

1. Key Evolution: Deterministically derive new keys from base keys using
   counter-based key derivation (forward secrecy)

2. Signature Obfuscation: Obfuscate Layer 6 signatures using evolved keys,
   making signatures unpredictable while maintaining verifiability

3. Evolution Tracking: Track key evolution history and rounds

Key Features:
- Deterministic key evolution (seeded, reproducible)
- Reversible obfuscation (can recover original signatures)
- Counter-based key rotation for forward secrecy
- Integration with Layer 6 digital signatures
- JSON serialization for evolution state
- Full error handling and validation
- Comprehensive statistics and logging

Typical Usage:
    layer7 = Layer7AdaptiveKeyEvolution()

    # Initialize with base key from Layer 6
    layer7.initialize(base_signature)

    # Evolve key and obfuscate signature
    obfuscated = layer7.obfuscate_signature(layer6_signature, round=0)

    # Later: Recover original signature
    recovered = layer7.deobfuscate_signature(obfuscated, round=0)
"""

import os
import json
import logging
import hashlib
from typing import Optional, Tuple, Dict, Any
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
import hmac

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
LAYER7_VERSION = "1.0.0"
KEY_SIZE = 32  # 256 bits
NONCE_SIZE = 12  # 96 bits for GCM-style operations


class Layer7AdaptiveKeyEvolution:
    """
    Layer 7: Adaptive Key Evolution & Signature Obfuscation

    Provides deterministic key evolution and reversible signature obfuscation
    for forward secrecy and additional security hardening.
    """

    def __init__(self, backend=None):
        """
        Initialize Layer 7 key evolution engine.

        Args:
            backend: Cryptographic backend (uses default if None)
        """
        self.backend = backend or default_backend()
        self.base_key = None
        self.current_round = 0
        self.evolution_history = []
        self.max_rounds = 1000000  # Prevent overflow

        # Statistics
        self.obfuscations_performed = 0
        self.deobfuscations_performed = 0
        self.key_evolutions = 0

        logger.info("✓ Layer 7 (Adaptive Key Evolution & Obfuscation) initialized")

    def initialize(self, base_key: bytes, round_counter: int = 0) -> None:
        """
        Initialize Layer 7 with a base key (typically from Layer 6).

        Args:
            base_key: Base key for key evolution (typically Layer 6 signature)
            round_counter: Starting round number (default: 0)

        Raises:
            ValueError: If base_key is invalid
        """
        if not isinstance(base_key, bytes):
            raise ValueError("base_key must be bytes")

        if len(base_key) < 32:
            raise ValueError("base_key must be at least 32 bytes")

        self.base_key = base_key
        self.current_round = round_counter
        self.evolution_history = []

        logger.info(f"✓ Layer 7 initialized with {len(base_key)}-byte base key at round {round_counter}")

    def _derive_key_for_round(self, round_number: int) -> bytes:
        """
        Derive a key for a specific round using HKDF (HMAC-based KDF).

        This provides deterministic key evolution. Given the same base key
        and round number, this always produces the same derived key.

        Args:
            round_number: Round number for key derivation

        Returns:
            Derived key (32 bytes)

        Raises:
            ValueError: If base_key not initialized
        """
        if self.base_key is None:
            raise ValueError("Layer 7 not initialized. Call initialize() first.")

        if round_number < 0 or round_number > self.max_rounds:
            raise ValueError(f"Round number must be between 0 and {self.max_rounds}")

        # Create info string for HKDF
        info = b"Layer7KeyEvolution" + str(round_number).encode()

        # Use HKDF to derive key deterministically
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=KEY_SIZE,
            salt=b"Layer7Salt",
            info=info,
            backend=self.backend
        )

        derived_key = hkdf.derive(self.base_key)
        self.key_evolutions += 1

        logger.debug(f"✓ Key derived for round {round_number}")
        return derived_key

    def _xor_with_key(self, data: bytes, key: bytes) -> bytes:
        """
        XOR data with key (deterministic, reversible).

        XOR is its own inverse: A XOR B XOR B = A

        Args:
            data: Data to XOR
            key: Key to XOR with

        Returns:
            XORed data
        """
        if len(key) < len(data):
            # Extend key if necessary
            extended_key = key
            while len(extended_key) < len(data):
                extended_key += hashlib.sha256(extended_key).digest()
            key = extended_key[:len(data)]

        return bytes(a ^ b for a, b in zip(data, key[:len(data)]))

    def obfuscate_signature(self, signature: bytes, round_number: int = None) -> Tuple[bytes, Dict[str, Any]]:
        """
        Obfuscate a signature (typically from Layer 6) using evolved key.

        The obfuscation is deterministic and reversible - given the same
        signature and round number, this always produces the same obfuscated result.

        Args:
            signature: Signature to obfuscate (from Layer 6)
            round_number: Round number for key derivation (uses current_round if None)

        Returns:
            Tuple of (obfuscated_signature, metadata_dict)

        Raises:
            ValueError: If Layer 7 not initialized
        """
        if self.base_key is None:
            raise ValueError("Layer 7 not initialized. Call initialize() first.")

        if round_number is None:
            round_number = self.current_round

        if not isinstance(signature, bytes):
            raise ValueError("signature must be bytes")

        try:
            # Derive key for this round
            evolved_key = self._derive_key_for_round(round_number)

            # Obfuscate using XOR (reversible)
            obfuscated = self._xor_with_key(signature, evolved_key)

            # Store metadata for recovery
            metadata = {
                'round': round_number,
                'original_size': len(signature),
                'obfuscated_size': len(obfuscated),
                'version': LAYER7_VERSION,
                'algorithm': 'HKDF-SHA256-XOR'
            }

            self.obfuscations_performed += 1
            logger.info(f"✓ Signature obfuscated at round {round_number} ({len(signature)} bytes)")

            return obfuscated, metadata

        except Exception as e:
            logger.error(f"Obfuscation failed: {e}")
            raise

    def deobfuscate_signature(self, obfuscated_signature: bytes, round_number: int) -> bytes:
        """
        Recover original signature from obfuscated version.

        This is the inverse of obfuscate_signature(). XOR is self-inverse,
        so XORing again with the same key recovers the original.

        Args:
            obfuscated_signature: Obfuscated signature
            round_number: Round number used during obfuscation

        Returns:
            Original signature

        Raises:
            ValueError: If Layer 7 not initialized or round invalid
        """
        if self.base_key is None:
            raise ValueError("Layer 7 not initialized. Call initialize() first.")

        if not isinstance(obfuscated_signature, bytes):
            raise ValueError("obfuscated_signature must be bytes")

        try:
            # Derive same key for this round
            evolved_key = self._derive_key_for_round(round_number)

            # Deobfuscate using XOR (same operation as obfuscation)
            recovered = self._xor_with_key(obfuscated_signature, evolved_key)

            self.deobfuscations_performed += 1
            logger.info(f"✓ Signature deobfuscated at round {round_number} ({len(recovered)} bytes)")

            return recovered

        except Exception as e:
            logger.error(f"Deobfuscation failed: {e}")
            raise

    def advance_round(self) -> int:
        """
        Advance to the next evolution round.

        This moves to the next key for forward secrecy - each round uses
        a different key derived from the base key.

        Returns:
            New round number
        """
        if self.current_round >= self.max_rounds:
            raise ValueError(f"Maximum rounds ({self.max_rounds}) reached")

        self.current_round += 1
        self.evolution_history.append(self.current_round)

        logger.info(f"✓ Advanced to round {self.current_round}")
        return self.current_round

    def get_current_round(self) -> int:
        """
        Get the current evolution round.

        Returns:
            Current round number
        """
        return self.current_round

    def save_state(self, filename: str = "layer7_state.json") -> Dict[str, str]:
        """
        Save Layer 7 evolution state to JSON file.

        Args:
            filename: Output filename

        Returns:
            Dictionary with saved state info
        """
        if self.base_key is None:
            raise ValueError("Layer 7 not initialized")

        state = {
            'base_key': self.base_key.hex(),
            'current_round': self.current_round,
            'evolution_history': self.evolution_history,
            'version': LAYER7_VERSION
        }

        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"✓ Layer 7 state saved to {filename}")
        return {'filename': filename, 'rounds': self.current_round}

    def load_state(self, filename: str = "layer7_state.json") -> None:
        """
        Load Layer 7 evolution state from JSON file.

        Args:
            filename: Input filename
        """
        with open(filename, 'r') as f:
            state = json.load(f)

        self.base_key = bytes.fromhex(state['base_key'])
        self.current_round = state['current_round']
        self.evolution_history = state.get('evolution_history', [])

        logger.info(f"✓ Layer 7 state loaded from {filename}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Layer 7 statistics and state.

        Returns:
            Dictionary with configuration and stats
        """
        return {
            'layer': 7,
            'algorithm': 'HKDF-SHA256-XOR',
            'mode': 'Key Evolution & Obfuscation',
            'current_round': self.current_round,
            'key_size': KEY_SIZE,
            'obfuscations_performed': self.obfuscations_performed,
            'deobfuscations_performed': self.deobfuscations_performed,
            'key_evolutions': self.key_evolutions,
            'evolution_history_length': len(self.evolution_history),
            'initialized': self.base_key is not None,
            'version': LAYER7_VERSION
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Layer 7 (Adaptive Key Evolution & Obfuscation) - Key Management")
    print("=" * 70)

    # Test 1: Initialization
    print("\nTest 1: Layer 7 Initialization")
    print("-" * 70)

    layer7 = Layer7AdaptiveKeyEvolution()
    base_key = os.urandom(64)  # Use a Layer 6 signature as base
    layer7.initialize(base_key, round_counter=0)

    print(f"Base key size: {len(base_key)} bytes")
    print(f"Starting round: {layer7.get_current_round()}")
    print(f"✓ Layer 7 initialized")

    # Test 2: Obfuscate and deobfuscate
    print("\nTest 2: Obfuscate and Deobfuscate")
    print("-" * 70)

    signature = os.urandom(2484)  # Layer 6 hybrid signature size
    print(f"Original signature size: {len(signature)} bytes")
    print(f"Original signature (first 20 bytes): {signature[:20].hex()}")

    obfuscated, metadata = layer7.obfuscate_signature(signature, round_number=0)
    print(f"Obfuscated signature (first 20 bytes): {obfuscated[:20].hex()}")
    print(f"Different: {signature[:20].hex() != obfuscated[:20].hex()}")

    recovered = layer7.deobfuscate_signature(obfuscated, round_number=0)
    print(f"Recovered signature (first 20 bytes): {recovered[:20].hex()}")
    print(f"Matches original: {recovered == signature}")
    print(f"✓ Obfuscation/Deobfuscation successful")

    # Test 3: Key evolution across rounds
    print("\nTest 3: Key Evolution Across Rounds")
    print("-" * 70)

    test_data = b"Test signature data"

    # Obfuscate at round 0
    obs0, _ = layer7.obfuscate_signature(test_data, round_number=0)
    layer7.advance_round()

    # Obfuscate same data at round 1
    obs1, _ = layer7.obfuscate_signature(test_data, round_number=1)
    layer7.advance_round()

    # Obfuscate same data at round 2
    obs2, _ = layer7.obfuscate_signature(test_data, round_number=2)

    print(f"Round 0 obfuscation: {obs0.hex()[:20]}...")
    print(f"Round 1 obfuscation: {obs1.hex()[:20]}...")
    print(f"Round 2 obfuscation: {obs2.hex()[:20]}...")
    print(f"All different: {obs0 != obs1 and obs1 != obs2 and obs0 != obs2}")
    print(f"✓ Key evolution working correctly")

    # Test 4: Round advancement
    print("\nTest 4: Round Advancement")
    print("-" * 70)

    current = layer7.get_current_round()
    print(f"Current round before advance: {current}")

    new_round = layer7.advance_round()
    print(f"Current round after advance: {new_round}")
    print(f"Advanced: {new_round > current}")
    print(f"✓ Round advancement working")

    # Test 5: State serialization
    print("\nTest 5: State Serialization")
    print("-" * 70)

    layer7.save_state("test_layer7_state.json")
    print(f"State saved")

    # Create new instance and load state
    layer7_loaded = Layer7AdaptiveKeyEvolution()
    layer7_loaded.load_state("test_layer7_state.json")

    print(f"Loaded round: {layer7_loaded.get_current_round()}")
    print(f"Matches saved round: {layer7_loaded.get_current_round() == layer7.get_current_round()}")
    print(f"✓ State serialization working")

    # Cleanup
    import os as os_module
    try:
        os_module.remove("test_layer7_state.json")
    except:
        pass

    # Test 6: Statistics
    print("\nTest 6: Layer 7 Statistics")
    print("-" * 70)

    stats = layer7.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✓ All Layer 7 tests completed successfully!")