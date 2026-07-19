#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# One-time provisioning of the `pch2r_dev` conda env to BUILD + RUN this project on
# DelftBlue (RHEL8: glibc 2.28, system gcc 8.5). The env ships with torch + nvcc but
# not a working C++/CUDA build toolchain, and some binaries were compiled on a newer
# OS (glibc 2.34) and must be rebuilt here. Run ONCE on a LOGIN node from the repo
# root (compiles need nvcc but no GPU):
#
#     conda activate pch2r_dev
#     bash examples/slurm/setup_cluster_env.sh
#
# Then submit training with `sbatch examples/slurm/train_rl.sbatch`.
# Re-runnable (rebuilds cleanly). See examples/slurm/README.md for the why.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

source /etc/profile.d/lmod.sh 2>/dev/null || true
source "$(conda info --base 2>/dev/null || echo /apps/generic/miniforge3/25.11.0)/etc/profile.d/conda.sh"
conda activate pch2r_dev
module load 2023r1 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true          # CUDA 12.1 headers/nvcc (matches torch cu121)
export CUDA_HOME="${CUDA_HOME:-/beegfs/apps/generic/cuda-12.1}"

echo "############ 1/5  runtime deps into pch2r_dev ############"
# sip=4.19: PyKDL is a SIP-4 binding. assimp: native lib for pyassimp (mesh loader).
# ninja: for the torch cpp_extension builds below.
conda install -y -c conda-forge 'sip=4.19.25' assimp ninja

echo "############ 2/5  gcc-11 toolchain in a SEPARATE env ############"
# pch2r_dev pins ld_impl=2.44, which blocks installing gcc-11 into it; put the
# compiler in its own env and use it via a PATH shim. CUDA 12.1 supports host gcc<=12.
if ! conda env list | grep -qE '^pn2build\s'; then
  conda create -y -n pn2build -c conda-forge 'gcc_linux-64=11' 'gxx_linux-64=11' 'binutils_linux-64=2.40'
fi
PN2="$HOME/.conda/envs/pn2build"; [ -x "$PN2/bin/x86_64-conda-linux-gnu-gcc" ] || PN2="$(conda info --base)/envs/pn2build"
GCC="$PN2/bin/x86_64-conda-linux-gnu-gcc"; GXX="$PN2/bin/x86_64-conda-linux-gnu-g++"
# shim: make plain gcc/g++ resolve to the conda gcc-11, adding /usr/include at lowest
# priority so Python.h's <crypt.h> resolves (the conda sysroot lacks it).
SHIM="$REPO/.gcc11-shim"; mkdir -p "$SHIM"
for n in gcc cc;  do printf '#!/bin/bash\nexec %s "$@" -idirafter /usr/include\n' "$GCC" > "$SHIM/$n"; chmod +x "$SHIM/$n"; done
for n in g++ c++; do printf '#!/bin/bash\nexec %s "$@" -idirafter /usr/include\n' "$GXX" > "$SHIM/$n"; chmod +x "$SHIM/$n"; done
export PATH="$SHIM:$PATH" CC=gcc CXX=g++ CUDAHOSTCXX=g++ TORCH_CUDA_ARCH_LIST=7.0   # V100 = sm_70

echo "############ 3/5  rebuild CUDA/C++ extensions (torch 2.4.1+cu121, sm_70) ############"
echo "--- pointnet2_ops (GA-DDPG PointNet++) ---"
( cd GA-DDPG/Pointnet2_PyTorch/pointnet2_ops_lib
  rm -rf build pointnet2_ops/_ext.cpython-38*.so *.egg-info
  TORCH_CUDA_ARCH_LIST=7.0 pip install -e . --no-build-isolation )
echo "--- omg_cuda (OMG-Planner SDF loss) ---"
( cd OMG-Planner/layers
  rm -rf build dist omg.egg-info
  TORCH_CUDA_ARCH_LIST=7.0 pip install . --no-build-isolation )
echo "--- CppYCBRenderer (OMG-Planner YCB renderer; was a glibc-2.34 binary) ---"
( cd OMG-Planner/ycb_render
  rm -rf build_cluster; mkdir build_cluster; cd build_cluster
  cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$(which python)" -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$REPO/OMG-Planner/ycb_render" \
    -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCUDA_HOST_COMPILER=gcc \
    -DCUDA_NVCC_FLAGS="-gencode=arch=compute_70,code=sm_70" >/dev/null
  make -j4 CppYCBRenderer )

echo "############ 4/5  fix table asset symlinks (pointed at the PC's pybullet_data) ############"
PBD="$CONDA_PREFIX/lib/python3.8/site-packages/pybullet_data/table"
TBL="$REPO/handover-sim/handover/data/assets/table"
for f in table.obj table.mtl table.png; do
  if [ -L "$TBL/$f" ] && [ ! -e "$TBL/$f" ]; then rm -f "$TBL/$f"; cp "$PBD/$f" "$TBL/$f"; fi
done

echo "############ 5/5  verify the built extensions import ############"
python - <<'PY'
import torch
import pointnet2_ops.pointnet2_modules   # noqa
import omg_cuda                           # noqa
import sys; sys.path.insert(0, "OMG-Planner/ycb_render")
import CppYCBRenderer                     # noqa
print("SETUP OK: torch", torch.__version__, "| pointnet2_ops, omg_cuda, CppYCBRenderer import")
PY
echo
echo "DONE. Launch training with:  sbatch examples/slurm/train_rl.sbatch"
