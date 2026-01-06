import tensorflow as tf
import torch

print("----- TensorFlow -----")
gpus_tf = tf.config.list_physical_devices('GPU')
if gpus_tf:
    print(f"TensorFlow sees {len(gpus_tf)} GPU(s):")
    for gpu in gpus_tf:
        print(f"  - {gpu}")
else:
    print("No GPU detected by TensorFlow.")

print("\n----- PyTorch -----")
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"PyTorch sees {num_gpus} GPU(s):")
    for i in range(num_gpus):
        print(f"  - {torch.cuda.get_device_name(i)}")
else:
    print("No GPU detected by PyTorch.")
