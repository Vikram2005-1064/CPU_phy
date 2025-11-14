# test_imports.py

try:
    from pack import ChaosEntropyGenerator, Layer2MLWE, generate_master_key
    print("Successfully imported ChaosEntropyGenerator, Layer2MLWE, and generate_master_key from pack")
except ImportError as e:
    print("Failed to import from pack:", e)

# Try using ChaosEntropyGenerator
try:
    layer1 = ChaosEntropyGenerator(num_balls=10, sim_frames=20, headless=True)
    entropy = layer1.extract_entropy()
    print("Layer1 entropy extraction successful, entropy keys:", entropy.keys())
except Exception as e:
    print("Error running ChaosEntropyGenerator:", e)

# Try Layer2MLWE usage
try:
    layer2 = Layer2MLWE()
    print("Layer2MLWE instance created successfully")
except Exception as e:
    print("Error creating Layer2MLWE:", e)

# Try generating a key with the helper
try:
    key_result = generate_master_key(num_balls=10, sim_frames=20)
    print("Generated master key:", key_result['master_key'][:16] + '...')
except Exception as e:
    print("Error generating master key:", e)
