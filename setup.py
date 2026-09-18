import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def _cuda_version_tuple():
    """Return the CUDA toolkit version reported by torch as a tuple."""
    if not torch.version.cuda:
        return ()
    try:
        return tuple(int(part) for part in torch.version.cuda.split("."))
    except ValueError:
        return ()


def _default_cuda_archs():
    """Keep old toolkits buildable while adding targets supported by newer nvcc."""
    cuda_version = _cuda_version_tuple()
    archs = ["70", "75", "80", "86"]
    if cuda_version >= (11, 8):
        archs.append("89")
    if cuda_version >= (12, 0):
        archs.append("90")
    if cuda_version >= (12, 8):
        archs.append("100")
    return archs


def _cuda_archs():
    value = os.getenv("BEVFUSION_CUDA_ARCHS")
    if not value:
        return _default_cuda_archs()

    archs = []
    for item in value.replace(",", ";").split(";"):
        item = item.strip().lower()
        if not item:
            continue
        for prefix in ("compute_", "sm_"):
            if item.startswith(prefix):
                item = item[len(prefix):]
        item = item.replace(".", "")
        if not item.isdigit() or len(item) not in (2, 3):
            raise ValueError(
                "Invalid BEVFUSION_CUDA_ARCHS entry {!r}; use values such as "
                "'8.6;8.9;9.0;10.0' or '86;89;90;100'".format(item)
            )
        if item not in archs:
            archs.append(item)
    if not archs:
        raise ValueError("BEVFUSION_CUDA_ARCHS did not contain an architecture")
    return archs


def _cuda_gencode_flags():
    return [
        "-gencode=arch=compute_{0},code=sm_{0}".format(arch)
        for arch in _cuda_archs()
    ]


def make_cuda_ext(
    name, module, sources, sources_cuda=[], extra_args=[], extra_include_path=[]
):

    define_macros = []
    extra_compile_args = {"cxx": [] + extra_args}

    if (torch.cuda.is_available() and torch.version.cuda is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        define_macros += [("WITH_CUDA", None)]
        extension = CUDAExtension
        extra_compile_args["nvcc"] = extra_args + [
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ] + _cuda_gencode_flags()
        sources += sources_cuda
    elif (torch.cuda.is_available() and torch.version.hip is not None) or os.getenv("FORCE_ROCM", "0") == 1:
        define_macros += [("WITH_ROCM", None)]
        extension = CUDAExtension
        extra_compile_args["hipcc"] = extra_args + [
            "-D__HIP_NO_HALF_OPERATORS__",
            "-D__HIP_NO_HALF_CONVERSIONS__",
            "-D__HIP_NO_HALF2_OPERATORS__",
        ]
        sources += sources_cuda
    else:
        print("Compiling {} without CUDA".format(name))
        extension = CppExtension

    return extension(
        name="{}.{}".format(module, name),
        sources=[os.path.join(*module.split("."), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
    )


if __name__ == "__main__":
    setup(
        name="mmdet3d",
        packages=find_packages(),
        include_package_data=True,
        package_data={"mmdet3d.ops": ["*/*.so"]},
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
        ],
        license="Apache License 2.0",
        ext_modules=[
            make_cuda_ext(
                name="sparse_conv_ext",
                module="mmdet3d.ops.spconv",
                extra_include_path=[
                    # PyTorch 1.5 uses ninjia, which requires absolute path
                    # of included files, relative path will cause failure.
                    os.path.abspath(
                        os.path.join(*"mmdet3d.ops.spconv".split("."), "include/")
                    )
                ],
                sources=[
                    "src/all.cc",
                    "src/reordering_cpu.cc",
                    "src/reordering_cuda.cu",
                    "src/indice_cpu.cc",
                    "src/indice_cuda.cu",
                    "src/maxpool_cpu.cc",
                    "src/maxpool_cuda.cu",
                ],
                extra_args=["-w", "-std=c++17"],
            ),
            make_cuda_ext(
                name="bev_pool_ext",
                module="mmdet3d.ops.bev_pool",
                sources=[
                    "src/bev_pool_cpu.cpp",
                    "src/bev_pool_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="iou3d_cuda",
                module="mmdet3d.ops.iou3d",
                sources=[
                    "src/iou3d.cpp",
                    "src/iou3d_kernel.cu",
                ],
            ),
            make_cuda_ext(
                name="voxel_layer",
                module="mmdet3d.ops.voxel",
                sources=[
                    "src/voxelization.cpp",
                    "src/scatter_points_cpu.cpp",
                    "src/scatter_points_cuda.cu",
                    "src/voxelization_cpu.cpp",
                    "src/voxelization_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="roiaware_pool3d_ext",
                module="mmdet3d.ops.roiaware_pool3d",
                sources=[
                    "src/roiaware_pool3d.cpp",
                    "src/points_in_boxes_cpu.cpp",
                ],
                sources_cuda=[
                    "src/roiaware_pool3d_kernel.cu",
                    "src/points_in_boxes_cuda.cu",
                ],
            ),
            make_cuda_ext(
                name="ball_query_ext",
                module="mmdet3d.ops.ball_query",
                sources=["src/ball_query_cpu.cpp"],
                sources_cuda=["src/ball_query_cuda.cu"],
            ),
            make_cuda_ext(
                name="knn_ext",
                module="mmdet3d.ops.knn",
                sources=["src/knn_cpu.cpp"],
                sources_cuda=["src/knn_cuda.cu"],
            ),
            make_cuda_ext(
                name="assign_score_withk_ext",
                module="mmdet3d.ops.paconv",
                sources=["src/assign_score_withk.cpp"],
                sources_cuda=["src/assign_score_withk_cuda.cu"],
            ),
            make_cuda_ext(
                name="group_points_ext",
                module="mmdet3d.ops.group_points",
                sources=["src/group_points_cpu.cpp"],
                sources_cuda=["src/group_points_cuda.cu"],
            ),
            make_cuda_ext(
                name="interpolate_ext",
                module="mmdet3d.ops.interpolate",
                sources=["src/interpolate.cpp"],
                sources_cuda=["src/three_interpolate_cuda.cu", "src/three_nn_cuda.cu"],
            ),
            make_cuda_ext(
                name="furthest_point_sample_ext",
                module="mmdet3d.ops.furthest_point_sample",
                sources=["src/furthest_point_sample_cpu.cpp"],
                sources_cuda=["src/furthest_point_sample_cuda.cu"],
            ),
            make_cuda_ext(
                name="gather_points_ext",
                module="mmdet3d.ops.gather_points",
                sources=["src/gather_points_cpu.cpp"],
                sources_cuda=["src/gather_points_cuda.cu"],
            ),
        ],
        cmdclass={"build_ext": BuildExtension},
        zip_safe=False,
    )
