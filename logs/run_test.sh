#!/bin/bash
cd /home/arhashemi/Workspace/welding/JAXTrace
source .venv/bin/activate
python -u example_workflow.py > logs/gpu_verification_test.log 2>&1 &
echo $!
