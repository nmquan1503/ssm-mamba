from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

this_dir = Path(__file__).parent
csrc_dir = this_dir / "csrc"

common_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ],
}

def make_extension(name, sources):
    return CUDAExtension(
        name=name,
        sources=[str(csrc_dir / s) for s in sources],
        include_dirs=[str(csrc_dir)],
        extra_compile_args=common_compile_args,
    )

setup(
    name="selective_ops",
    ext_modules=[
        make_extension(
            "selective_scan",
            [
                "selective_scan/bindings.cpp",
                "selective_scan/selective_scan.cpp",
                "selective_scan/forward_kernel.cu",
                "selective_scan/backward_kernel.cu",
            ],
        ),
        make_extension(
            "selective_update",
            [
                "selective_update/bindings.cpp",
                "selective_update/selective_update.cpp",
                "selective_update/kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)