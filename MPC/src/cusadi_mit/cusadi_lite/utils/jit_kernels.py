# CusADi (se-hwan/cusadi, wip 브랜치 40038882, MIT) 의 jit_kernels.py — 우리 쪽 수정 (Q&A 11/9):
#   1. -arch=sm_86 고정 → 지금 GPU 의 compute capability 로 (RTX 5080 = sm_120, nvcc 12.8 이상 필요)
#   2. 빌드 폴더를 cusadi_lite/ 안으로 (git 에 올리지 않음, .gitignore)
#   3. cuDSS 헤더·라이브러리를 pip 패키지 nvidia-cudss-cu12 (nvidia/cudss/{include,lib}) 에서 먼저 찾고, 없으면 CUDA_HOME.
#      pip 판에는 libcudss.so 심볼릭 링크가 없을 수 있어 실제 파일 이름으로 링크 (-l:libcudss.so.0) + rpath.
import glob
import importlib
import os
import sys

import torch
from torch.utils.cpp_extension import load

LITE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _arch_flags():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 가 없다 — CusADi 커널은 GPU 에서만 (노트북은 backend=casadi)")
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    print(f"\n********** CUDA architecture {major}.{minor} 용으로 컴파일 **********")
    return [f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"]


def compile_and_load_kernels(kernel_names):
    flags = _arch_flags()
    codegen_dir = os.path.join(LITE_DIR, "codegen")
    build_dir = os.path.join(LITE_DIR, "build_kernels")
    os.makedirs(build_dir, exist_ok=True)
    print("Loading JIT kernels from: ", codegen_dir)
    kernel_sources = [os.path.join(codegen_dir, f"{name}.cu") for name in sorted(kernel_names)]
    if not kernel_sources:
        print("No kernel sources found. Skipping JIT compilation.")
        return None
    kernel_sources.append(os.path.join(codegen_dir, "bindings.cpp"))
    module = load(name="cusadi_kernels",
                  sources=kernel_sources,
                  extra_cflags=["-O3"],
                  extra_cuda_cflags=["-O3", "--use_fast_math"] + flags,
                  verbose=False,
                  is_python_module=True,
                  build_directory=build_dir)
    sys.modules["cusadi_kernels"] = module
    importlib.invalidate_caches()
    return module


def find_cudss():
    """→ (include 폴더, lib 폴더, 링크 인자). pip nvidia-cudss-cu12 우선."""
    cands = []
    try:
        import nvidia.cudss as _c                                   # pip install nvidia-cudss-cu12
        for p in list(_c.__path__):
            cands.append((os.path.join(p, "include"), os.path.join(p, "lib")))
    except ImportError:
        pass
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cands += [(os.path.join(cuda_home, "include"), os.path.join(cuda_home, "lib64")),
              ("/usr/include", "/usr/lib/x86_64-linux-gnu")]
    for inc, lib in cands:
        if os.path.exists(os.path.join(inc, "cudss.h")):
            libs = sorted(glob.glob(os.path.join(lib, "libcudss.so*")))
            if not libs:
                continue
            plain = os.path.join(lib, "libcudss.so")
            name = "-lcudss" if os.path.exists(plain) else f"-l:{os.path.basename(libs[0])}"
            return inc, lib, [f"-L{lib}", name, f"-Wl,-rpath,{lib}"]
    raise RuntimeError("cudss.h / libcudss.so 를 못 찾음 — pip install nvidia-cudss-cu12 (CUDA 12 판)")


def compile_and_load_cudss():
    flags = _arch_flags()
    utils_dir = os.path.join(LITE_DIR, "utils")
    build_dir = os.path.join(LITE_DIR, "build_cudss")
    os.makedirs(build_dir, exist_ok=True)
    inc, lib, ld = find_cudss()
    print(f"cuDSS: include {inc}, lib {lib}")
    sources = [os.path.join(utils_dir, "cudss_binding.cpp"),
               os.path.join(utils_dir, "cudss_interface.cu"),
               os.path.join(utils_dir, "cuda_utils.cu")]
    module = load(name="cudss",
                  sources=sources,
                  extra_cflags=["-O3"],
                  extra_cuda_cflags=["-O3", "--use_fast_math"] + flags,
                  extra_include_paths=[inc],
                  extra_ldflags=ld,
                  verbose=False,
                  is_python_module=True,
                  build_directory=build_dir)
    sys.modules["cudss"] = module
    return module
