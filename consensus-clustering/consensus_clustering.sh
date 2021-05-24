#!/bin/sh
#SBATCH -J consensus_clustering
#SBATCH --mem=2gb
#SBATCH --ntasks=4
#SBATCH -o cons.out
#SBATCH -e cons.err

Rscript consensus_clustering_with_py.R
