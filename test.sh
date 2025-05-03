#!/bin/sh
#SBATCH -G 4
#SBATCH -C rtx4500
#SBATCH --mem=80g
cd /common/users/kll160/"Error Correction"/Error-Correcting
. "/common/users/kll160/hg/bin/activate"
vllm serve "./QwQ-32B" --trust-remote-code --tensor-parallel-size 4 --max_num_seqs 400 --max_num_batched_tokens 400 --gpu_memory_utilization .8 --max_model_len 400 --download_dir "/common/users/kll160/.cache" --host "localhost" --port 3465 --api-key "3857skg389dh"
