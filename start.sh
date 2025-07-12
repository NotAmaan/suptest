#!/bin/bash

echo "Starting container in $MODE_TO_RUN mode..."

case "$MODE_TO_RUN" in
    "pod")
        echo "Starting Gradio interface for interactive use..."
        # For pod mode, start the Gradio interface with --listen flag for RunPod
        python3 run_supir_gradio.py --listen --port 7860
        ;;
    
    "serverless")
        echo "Starting RunPod serverless worker..."
        # For serverless mode, start the RunPod handler
        python3 -u handler.py
        ;;
    
    *)
        echo "ERROR: Unknown mode '$MODE_TO_RUN'. Valid modes are: pod, serverless"
        exit 1
        ;;
esac