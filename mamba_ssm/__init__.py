'''__version__ = "2.3.2.post1"

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
'''

__version__ = "2.3.2.post1"

# Mamba-1 uses pure CUDA kernels and is always available.
from mamba_ssm.modules.mamba_simple import Mamba

# We build a list of missing dependencies to print one clean warning at the end.
_unavailable_modules = []

# Try to load Mamba-2 (requires Triton)
try:
    import triton
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:
    Mamba2 = None
    _unavailable_modules.append("Mamba-2 (requires Triton)")

# Try to load Mamba-3 (requires Triton + Tilelang)
try:
    from mamba_ssm.modules.mamba3 import Mamba3
except ImportError:
    Mamba3 = None
    _unavailable_modules.append("Mamba-3 (requires Triton + Tilelang)")

# Print a single, clear warning if we had to fall back.
if _unavailable_modules:
    print("\n" + "="*80)
    print(" [!] MAMBA-ON-CLUNKERS FALLBACK INITIATED")
    print(" [!] The following modules are unavailable on this platform:")
    for module in _unavailable_modules:
        print(f"     - {module}")
    print(" [!] Falling back to Mamba-1 (Pure CUDA). This is expected on legacy GPUs.")
    print("="*80 + "\n")

__all__ = ["Mamba", "Mamba2", "Mamba3"]