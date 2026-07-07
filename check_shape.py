import tensorflow as tf
from pathlib import Path

# Patch Keras Dense layer to handle quantization_config serialization issue in Keras 3
try:
    original_from_config = tf.keras.layers.Dense.from_config
    @classmethod
    def patched_from_config(cls, config):
        if "quantization_config" in config:
            config.pop("quantization_config")
        if "config" in config and isinstance(config["config"], dict):
            config["config"].pop("quantization_config", None)
        return original_from_config(config)
    tf.keras.layers.Dense.from_config = patched_from_config
    print("Successfully patched Keras Dense.from_config")
except Exception as e:
    print(f"Could not patch Dense.from_config: {e}")

class CompatDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        if 'quantization_config' in config:
            config.pop('quantization_config')
        if 'config' in config and isinstance(config['config'], dict):
            config['config'].pop('quantization_config', None)
        return super().from_config(config)

models_dir = Path('models')
for p in models_dir.glob('*'):
    if p.suffix in ['.h5', '.keras']:
        print(f"\nChecking model: {p.name}")
        try:
            model = tf.keras.models.load_model(str(p), custom_objects={'Dense': CompatDense})
            print("  EXPECTED_SHAPE:", model.input_shape)
            print("  OUTPUT_SHAPE:", model.output_shape)
        except Exception as e:
            print("  Failed to load:", e)

