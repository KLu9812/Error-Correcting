#!/bin/sh
#SBATCH -G 4
#SBATCH -C rtx4500
#SBATCH --mem=100g
cd /common/users/kll160/"Error Correction"/Error-Correcting
. "/common/users/kll160/hg/bin/activate"
python vscopcv.py
