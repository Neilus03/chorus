#!/bin/bash
#SBATCH --job-name=chorus_data_gen
#SBATCH --output=logs/data_gen_%j.out
#SBATCH --error=logs/data_gen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --time=48:00:00

# 1. Load the core Euler modules (Make sure Python is 3.12.8!)
module load stack/2024-06 gcc/12.2.0 python/3.12.8 cuda/12.4.1

# 2. Activate your Python environment from the WORK drive
source /cluster/work/igp_psr/nedela/litept-env/bin/activate

# 3. Move to your PERSONAL drive
cd ~/nedela/projects/chorus/chorus

# 4. --- THE GOLDEN FIXES ---
# Prevent network storage deadlocks
export PYTHONDONTWRITEBYTECODE=1
# Prevent OpenBLAS/NumPy from requesting 256 cores and freezing
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "STEP 1: Starting Python on the Compute Node..."

# 5. Run the data generation pipeline
python -u scripts/run_streaming.py
