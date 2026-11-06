"""MKL must settle its vector-math kernels before threads can race for them. See src/__init__.py."""
import ctypes
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LIBRARY = str(Path(torch.__file__).parent / "lib" / "libtorch_cpu.so")

# MKL reads this variable only when it first chooses, then keeps its choice. So asking for a kernel
# set after the import and getting it means the import left the choice open for threads to race.
PROBE = f"""
import ctypes, os
import src
os.environ["MKL_VML_DEBUG_CPU_TYPE"] = "1"
print(ctypes.CDLL({LIBRARY!r}).mkl_vml_serv_cpu_detect())
"""


def can_probe():
    try:
        return hasattr(ctypes.CDLL(LIBRARY), "mkl_vml_serv_cpu_detect")
    except OSError:
        return False


@pytest.mark.skipif(not can_probe(), reason="This PyTorch has no MKL vector math to probe")
def test_importing_the_package_settles_mkl_kernels():
    result = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert int(result.stdout) != 1
