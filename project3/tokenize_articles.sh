#!/bin/bash
#SBATCH -J bd-project4
#SBATCH --partition=main
#SBATCH --nodes=1                 # node count
#SBATCH --ntasks=1                # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G
#SBATCH --time=47:59:59           # total run time limit (HH:MM:SS)

module load openjdk/1.8.0_265-b01
module load python/3.7.7
source venv/bin/activate
 
lscpu
free -g -h -t
echo '# nodes: ----------------'
printenv SLURM_NNODES

python tokenize_articles.py ../chunks
