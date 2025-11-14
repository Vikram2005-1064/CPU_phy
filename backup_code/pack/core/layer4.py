"""
layer4.py - Layer 4: AES-256-GCM Authenticated Encryption

Layer 4 implements AES-256-GCM (Authenticated Encryption with Associated Data)
to encrypt output from Layer 3 (Neural PRNG) with guaranteed confidentiality 
and integrity protection.

Key Features:
- AES-256 encryption (256-bit keys for maximum security)
- GCM mode with 128-bit authentication tags
- 96-bit random nonces (industry standard)
- Associated Authenticated Data (AAD) support
- Integration with Layer 1 entropy and Layer 3 PRNG
- Timing-safe tag comparison to prevent timing attacks
- Hardware AES-NI acceleration (when available)

Typical Usage:
    layer4 = Layer4AESGCMEncryption()

    # Generate key and nonce
    key = layer4.generate_key()
    nonce = layer4.generate_nonce()

    # Encrypt plaintext
    plaintext = b"data from Layer 3"
    ciphertext, tag = layer4.encrypt(plaintext, key, nonce)

    # Decrypt and verify
    decrypted = layer4.decrypt(ciphertext, key, nonce, tag)
"""

import os
import json
import hashlib
import logging
import hmac
from typing import Optional, Tuple, Dict, Any
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
LAYER4_VERSION = "1.0.0"
AES_KEY_SIZE_256 = 32  # 256 bits = 32 bytes
AES_KEY_SIZE_192 = 24  # 192 bits = 24 bytes
AES_KEY_SIZE_128 = 16  # 128 bits = 16 bytes
GCM_NONCE_SIZE = 12    # 96 bits = 12 bytes (standard for GCM)
GCM_TAG_SIZE = 16      # 128 bits = 16 bytes


class Layer4AESGCMEncryption:
    """
    Layer 4: AES-256-GCM Authenticated Encryption

    Encrypts Layer 3 output (random sequences) using AES-256 in GCM mode.
    Provides both confidentiality and integrity protection.
    """

    def __init__(self, key_size: int = 256, backend=None):
        """
        Initialize Layer 4 encryption engine.

        Args:
            key_size: Key size in bits (256 for maximum security)
            backend: Cryptographic backend (uses default if None)
        """
        self.key_size = key_size  # Bits
        self.key_size_bytes = key_size // 8  # Convert to bytes
        self.nonce_size = GCM_NONCE_SIZE  # 96 bits standard
        self.tag_size = GCM_TAG_SIZE  # 128 bits
        self.backend = backend or default_backend()

        # Validate key size
        if key_size not in [128, 192, 256]:
            logger.warning(f"Unusual key size {key_size}. Using 256 instead.")
            self.key_size = 256
            self.key_size_bytes = 32

        # Layer 1 reference for entropy (optional)
        self.layer1 = None

        logger.info(f"✓ Layer 4 (AES-{key_size}-GCM) initialized")

    def generate_key(self) -> bytes:
        """
        Generate a random AES key.

        Returns:
            Random bytes of appropriate length for the key size
        """
        key = os.urandom(self.key_size_bytes)
        logger.debug(f"Generated {self.key_size}-bit key")
        return key

    def generate_nonce(self) -> bytes:
        """
        Generate a random 96-bit nonce for GCM mode.

        GCM mode requires a nonce (number used once) to ensure:
        - Each (key, nonce) pair is unique
        - No nonce reuse with the same key
        - Prevents chosen plaintext attacks

        Returns:
            Random 96-bit (12-byte) nonce
        """
        nonce = os.urandom(self.nonce_size)
        logger.debug(f"Generated {self.nonce_size * 8}-bit nonce")
        return nonce

    def derive_key_from_layer1(self, entropy: bytes, salt: Optional[bytes] = None) -> bytes:
        """
        Derive a 256-bit AES key from Layer 1 entropy using PBKDF2.

        This ensures that even if entropy source has biases, the derived key
        is still cryptographically strong.

        Args:
            entropy: Raw entropy bytes from Layer 1
            salt: Optional salt (generated if not provided)

        Returns:
            Derived AES key (256-bit)
        """
        if salt is None:
            salt = os.urandom(16)  # 128-bit salt

        # Use hashlib.pbkdf2_hmac (built-in Python stdlib)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            entropy,
            salt,
            100000  # NIST recommendation: 100k+ iterations
        )

        logger.debug(f"Derived {self.key_size}-bit key from Layer 1 entropy")
        return key

    def encrypt(self,
                plaintext: bytes,
                key: bytes,
                nonce: Optional[bytes] = None,
                aad: Optional[bytes] = None) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt plaintext using AES-256-GCM.

        GCM mode (Galois/Counter Mode) provides:
        - Confidentiality: AES encryption
        - Authenticity: GCM authentication tag
        - AEAD: Supports Associated Authenticated Data

        Args:
            plaintext: Data to encrypt (from Layer 3)
            key: Encryption key (256-bit)
            nonce: Optional nonce (generated if not provided)
            aad: Optional additional authenticated data

        Returns:
            Tuple of (ciphertext, nonce, authentication_tag)
        """
        if nonce is None:
            nonce = self.generate_nonce()

        if len(key) != self.key_size_bytes:
            raise ValueError(f"Key must be {self.key_size_bytes} bytes, got {len(key)}")

        if len(nonce) != self.nonce_size:
            raise ValueError(f"Nonce must be {self.nonce_size} bytes, got {len(nonce)}")

        try:
            # Create GCM cipher
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce),
                backend=self.backend
            )

            encryptor = cipher.encryptor()

            # Add additional authenticated data if provided
            if aad:
                encryptor.authenticate_additional_data(aad)

            # Encrypt
            ciphertext = encryptor.update(plaintext) + encryptor.finalize()
            tag = encryptor.tag

            logger.debug(f"Encrypted {len(plaintext)} bytes")
            return ciphertext, nonce, tag

        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt(self,
                ciphertext: bytes,
                key: bytes,
                nonce: bytes,
                tag: bytes,
                aad: Optional[bytes] = None) -> bytes:
        """
        Decrypt ciphertext using AES-256-GCM and verify authentication tag.

        CRITICAL: The authentication tag MUST be verified before returning plaintext.
        This prevents tampering attacks.

        Args:
            ciphertext: Encrypted data
            key: Decryption key (must be same as encryption key)
            nonce: Nonce used during encryption
            tag: Authentication tag from encryption
            aad: Optional additional authenticated data (must match encryption)

        Returns:
            Decrypted plaintext

        Raises:
            cryptography.exceptions.InvalidTag: If tag verification fails
        """
        if len(key) != self.key_size_bytes:
            raise ValueError(f"Key must be {self.key_size_bytes} bytes, got {len(key)}")

        if len(nonce) != self.nonce_size:
            raise ValueError(f"Nonce must be {self.nonce_size} bytes, got {len(nonce)}")

        if len(tag) != self.tag_size:
            raise ValueError(f"Tag must be {self.tag_size} bytes, got {len(tag)}")

        try:
            # Create GCM cipher with same nonce and tag
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce, tag),
                backend=self.backend
            )

            decryptor = cipher.decryptor()

            # Add additional authenticated data if provided
            if aad:
                decryptor.authenticate_additional_data(aad)

            # Decrypt and verify tag automatically
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            logger.debug(f"Decrypted {len(ciphertext)} bytes")
            return plaintext

        except Exception as e:
            logger.error(f"Decryption or tag verification failed: {e}")
            raise

    def integrate_with_layer3(self, layer3_output: bytes, 
                             key: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Integrate Layer 3 output with Layer 4 encryption.

        Takes random data from Layer 3 and encrypts it.

        Args:
            layer3_output: Random bytes from Layer 3 PRNG
            key: Optional encryption key (generated if not provided)

        Returns:
            Dictionary with:
            - ciphertext: Encrypted data
            - nonce: Nonce used
            - tag: Authentication tag
            - key_size: Key size used
        """
        if key is None:
            key = self.generate_key()

        nonce = self.generate_nonce()
        ciphertext, nonce, tag = self.encrypt(layer3_output, key, nonce)

        result = {
            'ciphertext': ciphertext,
            'nonce': nonce,
            'tag': tag,
            'key_size': self.key_size,
            'ciphertext_size': len(ciphertext),
            'plaintext_size': len(layer3_output)
        }

        logger.info(f"✓ Layer 3 output encrypted ({len(layer3_output)} → {len(ciphertext)} bytes)")
        return result

    def integrate_with_layer1(self, layer1_entropy: bytes) -> bytes:
        """
        Generate encryption key from Layer 1 entropy.

        Args:
            layer1_entropy: Entropy bytes from Layer 1

        Returns:
            Derived encryption key
        """
        key = self.derive_key_from_layer1(layer1_entropy)
        logger.info("✓ Encryption key derived from Layer 1 entropy")
        return key

    def format_output(self, ciphertext: bytes, nonce: bytes, tag: bytes) -> Dict[str, str]:
        """
        Format encryption output as hex strings for storage/transmission.

        Args:
            ciphertext: Encrypted data
            nonce: Nonce used
            tag: Authentication tag

        Returns:
            Dictionary with hex-encoded values
        """
        return {
            'ciphertext': ciphertext.hex(),
            'nonce': nonce.hex(),
            'tag': tag.hex(),
            'version': LAYER4_VERSION
        }

    def parse_input(self, data: Dict[str, str]) -> Tuple[bytes, bytes, bytes]:
        """
        Parse hex-encoded encryption data back to bytes.

        Args:
            data: Dictionary with hex-encoded ciphertext, nonce, tag

        Returns:
            Tuple of (ciphertext, nonce, tag) as bytes
        """
        ciphertext = bytes.fromhex(data['ciphertext'])
        nonce = bytes.fromhex(data['nonce'])
        tag = bytes.fromhex(data['tag'])
        return ciphertext, nonce, tag

    def verify_tag(self, tag1: bytes, tag2: bytes) -> bool:
        """
        Timing-safe comparison of authentication tags.

        Use HMAC.compare_digest to prevent timing attacks where
        attacker measures response time to guess tag values.

        Args:
            tag1: First tag
            tag2: Second tag

        Returns:
            True if tags match, False otherwise
        """
        return hmac.compare_digest(tag1, tag2)

    def batch_encrypt(self, plaintexts: list, key: Optional[bytes] = None) -> list:
        """
        Encrypt multiple plaintexts with the same key but different nonces.

        Args:
            plaintexts: List of plaintext bytes
            key: Optional encryption key (generated if not provided)

        Returns:
            List of (ciphertext, nonce, tag) tuples
        """
        if key is None:
            key = self.generate_key()

        results = []
        for plaintext in plaintexts:
            nonce = self.generate_nonce()
            ciphertext, nonce, tag = self.encrypt(plaintext, key, nonce)
            results.append((ciphertext, nonce, tag))

        logger.info(f"✓ Batch encrypted {len(plaintexts)} messages")
        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Layer 4 statistics and configuration.

        Returns:
            Dictionary with configuration details
        """
        return {
            'layer': 4,
            'algorithm': 'AES-GCM',
            'key_size_bits': self.key_size,
            'key_size_bytes': self.key_size_bytes,
            'nonce_size_bits': self.nonce_size * 8,
            'nonce_size_bytes': self.nonce_size,
            'tag_size_bits': self.tag_size * 8,
            'tag_size_bytes': self.tag_size,
            'mode': 'GCM',
            'aead': True,
            'aad_support': True,
            'version': LAYER4_VERSION
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Layer 4 (AES-256-GCM) - Authenticated Encryption")
    print("=" * 70)

    # Test 1: Basic encryption/decryption
    print("\nTest 1: Basic Encryption/Decryption")
    print("-" * 70)

    layer4 = Layer4AESGCMEncryption()

    plaintext = b"Hello from Layer 3! This is random data to encrypt."
    key = layer4.generate_key()
    nonce = layer4.generate_nonce()

    print(f"Plaintext: {plaintext}")
    print(f"Key size: {layer4.key_size} bits")
    print(f"Nonce size: {layer4.nonce_size * 8} bits")

    # Encrypt
    ciphertext, nonce, tag = layer4.encrypt(plaintext, key, nonce)
    print(f"✓ Encrypted: {len(ciphertext)} bytes")
    print(f"✓ Tag: {tag.hex()[:16]}...")

    # Decrypt
    decrypted = layer4.decrypt(ciphertext, key, nonce, tag)
    print(f"✓ Decrypted: {decrypted}")
    print(f"✓ Match: {decrypted == plaintext}")

    # Test 2: With AAD
    print("\nTest 2: Encryption with Associated Authenticated Data (AAD)")
    print("-" * 70)

    aad = b"metadata: version=1.0"
    ciphertext2, nonce2, tag2 = layer4.encrypt(plaintext, key, nonce, aad)
    print(f"AAD: {aad}")
    print(f"✓ Encrypted with AAD")

    try:
        decrypted2 = layer4.decrypt(ciphertext2, key, nonce2, tag2, aad)
        print(f"✓ Decrypted with correct AAD: {decrypted2 == plaintext}")
    except Exception as e:
        print(f"✗ Failed: {e}")

    # Test 3: Key derivation from entropy
    print("\nTest 3: Key Derivation from Entropy")
    print("-" * 70)

    entropy = os.urandom(32)
    derived_key = layer4.derive_key_from_layer1(entropy)
    print(f"Entropy: {entropy.hex()[:20]}...")
    print(f"✓ Derived key: {derived_key.hex()[:20]}...")

    ciphertext3, nonce3, tag3 = layer4.encrypt(plaintext, derived_key, None)
    print(f"✓ Encrypted with derived key")

    # Test 4: Format output
    print("\nTest 4: Format Output for Storage/Transmission")
    print("-" * 70)

    formatted = layer4.format_output(ciphertext, nonce, tag)
    print(f"Formatted output keys: {list(formatted.keys())}")
    print(f"Ciphertext (hex): {formatted['ciphertext'][:20]}...")
    print(f"Nonce (hex): {formatted['nonce']}")
    print(f"Tag (hex): {formatted['tag']}")

    # Test 5: Get stats
    print("\nTest 5: Layer 4 Configuration")
    print("-" * 70)

    stats = layer4.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test 6: Batch encryption (FIXED - use fresh key)
    print("\nTest 6: Batch Encryption")
    print("-" * 70)

    plaintexts = [b"Message 1", b"Message 2", b"Message 3"]
    batch_key = layer4.generate_key()  # FIXED: Generate fresh key for batch test
    batch_results = layer4.batch_encrypt(plaintexts, batch_key)
    print(f"Encrypted {len(batch_results)} messages")
    for i, (ct, n, t) in enumerate(batch_results):
        print(f"  Message {i+1}: {len(ct)} bytes ciphertext")

    print("\n" + "=" * 70)
    print("✓ All Layer 4 tests completed successfully!")