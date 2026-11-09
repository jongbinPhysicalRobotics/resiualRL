"""cusadi_lite — CusADi (Se Hwan Jeon, https://github.com/se-hwan/cusadi, MIT 라이선스) 에서 우리 MPC 에 필요한 부분만.

가져온 것 (wip 브랜치 40038882, 2026-04-01): parallelization/ 의 kernel_codegen · kernel_operations · cusadi_function ·
parallelize_functions · utils/{jit_kernels, cuda_utils.cu, cudss_interface.{cu,h}, cudss_binding.cpp}.
가져오지 않은 것: pinocchio 모델 · 최적화 문제 틀 · ADMM backend (우리 ADMM 은 batch_mpc.BatchOSQP — OSQP 1.1.3 재현).
수정: utils/jit_kernels.py 머리말 참고 (GPU 아키텍처 자동, 빌드 폴더, pip cuDSS), kernel_codegen.py (codegen 폴더 자동 생성).
생성물 (codegen/ · build_kernels/ · build_cudss/) 은 git 에 올리지 않는다.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def parallelize(fns, batch_size, precision="double", dynamic_batching=False):
    """ca.Function 목록 → {이름: CusadiFunction}. 커널 코드 생성 (바뀐 것만) + JIT 컴파일."""
    from .parallelize_functions import parallelize_functions
    return parallelize_functions(fns, batch_size=batch_size, precision=precision, dynamic_batching=dynamic_batching)


def load_cudss():
    """cuDSS 연결부 (cudssInterface_float / _double) JIT 컴파일 · import."""
    from .utils.jit_kernels import compile_and_load_cudss
    return compile_and_load_cudss()
