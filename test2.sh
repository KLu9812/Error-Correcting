#!/bin/sh
cd /common/users/kll160/"Error Correction"/Error-Correcting
. "/common/users/kll160/hg2/bin/activate"
python3 run.py
python3 -m vllm.entrypoints.openai.run_batch -i "prompts.jsonl" -o "results.jsonl" --model "./QwQ-32B" --trust-remote-code --dtype "float16" --tensor-parallel-size 8 --download_dir "/common/users/kll160/.cache"
python3 run2.py
