# Reproducibility Guidelines

The `ReproducibilityManager` ([pathoai/experiments/reproducibility.py](file:///d:/Research/PathoAI-Platform/pathoai/experiments/reproducibility.py)) sets master PRNG seeds across Python `random`, `numpy.random`, and `torch.manual_seed`, enforcing deterministic CUDA execution (`torch.backends.cudnn.deterministic = True`).
