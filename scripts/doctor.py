from __future__ import annotations
import os, platform, sys

def main():
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    try:
        import torch
        print("PyTorch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            props = torch.cuda.get_device_properties(0)
            print("GPU memory GiB:", round(props.total_memory / 1024**3, 2))
    except Exception as e:
        print("PyTorch: ERROR", repr(e))
    for pkg in ["transformers", "datasets", "accelerate"]:
        try:
            mod = __import__(pkg)
            print(f"{pkg}:", getattr(mod, "__version__", "unknown"))
        except Exception as e:
            print(f"{pkg}: ERROR", repr(e))
    for name in ["SPNER_DATA_ROOT", "SPNER_ARTIFACT_ROOT", "HF_HOME"]:
        print(f"{name}:", os.environ.get(name, "<not set>"))

if __name__ == "__main__":
    main()
