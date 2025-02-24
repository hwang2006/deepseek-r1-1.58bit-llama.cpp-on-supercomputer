#!/bin/bash
#SBATCH --comment=pytorch
#SBATCH --partition=amd_a100nv_8
##SBATCH --partition=eme_h200nv_8
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

# ============================
# Clean Up Old Port Forwarding
# ============================
rm -f port_forwarding_command

# Generate a Random Port (Avoiding Conflicts)
PORT_JU=$(($RANDOM % 10000 + 10000)) # Ensures port is between 10000–20000
SERVER=$(hostname)
PORT_JU=8080

echo "Using port: $PORT_JU on server: $SERVER"

# Create SSH Port Forwarding Command
PORT_FORWARD_CMD="ssh -L localhost:8080:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
echo "$PORT_FORWARD_CMD" > port_forwarding_command
echo "To access the web UI, run the following command on your local machine:"
echo "$PORT_FORWARD_CMD"

# ============================
# Load Modules
# ============================
echo "Loading required modules..."
module load gcc/10.2.0 cuda/12.1 cmake/3.26.2

# ============================
# Function: Start Llama.cpp Server
# ============================
start_llama_server() {
  echo "🔍 Detecting available GPUs..."
  
  # Get the number of GPUs
  NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

  # Set CUDA_VISIBLE_DEVICES dynamically
  if [ "$NUM_GPUS" -eq 1 ]; then
      export CUDA_VISIBLE_DEVICES=0
  elif [ "$NUM_GPUS" -eq 2 ]; then
      export CUDA_VISIBLE_DEVICES=0,1
  fi

  echo "✅ Detected $NUM_GPUS GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  # Detect GPU type
  GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)

  # Determine the correct llama-server binary path
  if [[ "$GPU_TYPE" == *"A100"* ]]; then
      LLAMA_SERVER_BIN="/scratch/$USER/llama.cpp/llama.cpp.a100/build/bin/llama-server"
      if [ "$NUM_GPUS" -eq 1 ]; then
          N_GPU_LAYERS=24
      else
          N_GPU_LAYERS=40
      fi
  elif [[ "$GPU_TYPE" == *"H200"* ]]; then
      LLAMA_SERVER_BIN="/scratch/$USER/llama.cpp/llama.cpp.h200/build/bin/llama-server"
      if [ "$NUM_GPUS" -eq 1 ]; then
          N_GPU_LAYERS=50
      else
          N_GPU_LAYERS=62
      fi
  else
      echo "⚠️ Unknown GPU type: $GPU_TYPE. Using default settings."
      LLAMA_SERVER_BIN="/scratch/$USER/llama.cpp/llama.cpp.default/build/bin/llama-server"
      N_GPU_LAYERS=20  # Default fallback value
  fi

  echo "🚀 GPU Type: $GPU_TYPE, Using server binary: $LLAMA_SERVER_BIN"
  echo "🎯 Setting n-gpu-layers to: $N_GPU_LAYERS"

  # Llama server settings
  LLAMA_MODEL="/scratch/qualis/llama.cpp/DeepSeek-R1-GGUF/DeepSeek-R1-UD-IQ1_S/DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf"
  LLAMA_LOG="llama_server.log"
  LLAMA_PORT=10000

  # Kill any existing llama-server process
  echo "🛑 Stopping any existing Llama.cpp server..."
  pkill -f "llama-server" || true

  # Start Llama.cpp server
  echo "🚀 Starting Llama.cpp server..."
  $LLAMA_SERVER_BIN --model $LLAMA_MODEL --port $LLAMA_PORT --ctx-size 8194 --n-gpu-layers $N_GPU_LAYERS > "$LLAMA_LOG" 2>&1 &

  sleep 2  # Allow some time for the server to start

  # Verify if the server started correctly
  if ! pgrep -f "llama-server" > /dev/null; then
    echo "❌ Llama.cpp server failed to start! Check $LLAMA_LOG for errors."
    exit 1
  else
    echo "✅ Llama.cpp server is running on port $LLAMA_PORT"
  fi
}


# ============================
# Function: Start Open-WebUI
# ============================
start_open_webui() {
  echo "Starting Open-WebUI..."

  source ~/.bashrc
  conda activate llama.cpp

  WEBUI_LOG="webui.log"

  # Kill any existing Open-WebUI process
  pkill -f "open-webui serve" || true

  # Run Open-WebUI in background
  open-webui serve --port $PORT_JU > "$WEBUI_LOG" 2>&1 

  # Verify if Open-WebUI started correctly
  if ! pgrep -f "open-webui serve" > /dev/null; then
    echo "❌ Open-WebUI failed to start! Check $WEBUI_LOG for errors."
    exit 1
  else
    echo "✅ Open-WebUI is running on port $PORT_JU"
  fi
}

# ============================
# Start Both Servers
# ============================
start_llama_server
start_open_webui

#echo "🎉 Servers started successfully!"
#echo "Llama.cpp logs: tail -f llama_server.log"
#echo "Open-WebUI logs: tail -f webui.log"
#echo "To access Open-WebUI, run the SSH command saved in 'port_forwarding_command'."

