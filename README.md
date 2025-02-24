# Running DeepSeek-R1 Dynamic 1.58-bit with Llama.cpp on a Supercomputer

DeepSeek-R1, the recently released AI reasoning model from the Chinese AI startup DeepSeek, has gained significant attention for its performance, comparable to leading models like OpenAI's o1 reasoning model. It is open-source and free to use, allowing users to download, modify, and run it for their own purposes.

This repository demonstrates how to run and test DeepSeek-R1 in its dynamic 1.58-bit quantized form using Llama.cpp on a SLURM-managed supercomputer. Thanks to advanced quantization techniques, the full 671B parameter model is compressed to just 131GB, making it significantly more accessible. You can now run it efficiently on a supercomputer with 1 or 2 A100 or H200 GPUs allocated through your account, **with a two H200 GPUs setup mostly recommended for optimal performance**. This removes the need for extremely large GPU configurations.

Llama.cpp provides an efficient framework for running large-scale AI models on CPUs and GPUs with optimized inference. This guide walks you through the steps needed to set up, deploy, and interact with DeepSeek-R1 in its quantized form on a high-performance computing (HPC) environment.

**Features:**

*   **Dynamic 1.58-bit quantization**: Reduces memory usage while maintaining model performance.
*   **Supercomputer compatibility**: Run DeepSeek-R1 efficiently on SLURM-managed HPC clusters.
*   **llama.cpp integration**: Utilize the optimized inference engine for fast execution.
*   **Supports 1-2 GPUs**: Can be run on a supercomputer with 1 or 2 A100 or H200 GPUs using individual user accounts.

**Contents**
* [KISTI Neuron GPU Cluster](#kisti-neuron-gpu-cluster)
* [Installing Conda](#installing-conda)
* [Installing Llama.cpp](#installing-llama.cpp)
* [Cloning the Repository](#cloning-the-repository)
* [Download the dynamic quantized version of DeepSeek-R1](#download-the-dynamic-quantized-version-of-deepSeek-r1)
* [Running Open WebUI](#running-open-webui)
* [Reference](#reference)

## KISTI Neuron GPU Cluster
Neuron is a KISTI GPU cluster system consisting of 65 nodes with 300 GPUs (40 of NVIDIA H200 GPUs, 120 of NVIDIA A100 GPUs and 140 of NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205237254-b916eccc-e4b7-46a8-b7ba-c156e7609314.png"/></p>

## Installing Conda
Once logging in to Neuron, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

1. Check the Neuron system specification
```
[glogin01]$ cat /etc/*release*
CentOS Linux release 7.9.2009 (Core)
Derived from Red Hat Enterprise Linux 7.8 (Source)
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"

CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"

CentOS Linux release 7.9.2009 (Core)
CentOS Linux release 7.9.2009 (Core)
cpe:/o:centos:centos:7
```

2. Download Anaconda or Miniconda. Miniconda comes with python, conda (package & environment manager), and some basic packages. Miniconda is fast to install and could be sufficient for distributed deep learning training practices. 
```
# (option 1) Anaconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh --no-check-certificate
```
```
# (option 2) Miniconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate
```

3. Install Miniconda. By default conda will be installed in your home directory, which has a limited disk space. You will install and create subsequent conda environments on your scratch directory. 
```
[glogin01]$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
[glogin01]$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>                               <======== press ENTER here
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes                      <========= type yes here 

Miniconda3 will now be installed into this location:
/home01/qualis/miniconda3        

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home01/qualis/miniconda3] >>> /scratch/$USER/miniconda3  <======== type /scratch/$USER/miniconda3 here
PREFIX=/scratch/qualis/miniconda3
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3
.
.
.
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish to update your shell profile to automatically initialize conda?
This will activate conda on startup and change the command prompt when activated.
If you'd prefer that conda's base environment not be activated on startup,
   run the following command when conda is activated:

conda config --set auto_activate_base false

You can undo this by running `conda init --reverse $SHELL`? [yes|no]
[no] >>> yes         <========== type yes here
.
.
.
no change     /scratch/qualis/miniconda3/etc/profile.d/conda.csh
modified      /home01/qualis/.bashrc

==> For changes to take effect, close and re-open your current shell. <==

Thank you for installing Miniconda3!
```

4. finalize installing Miniconda with environment variables set including conda path

```
[glogin01]$ source ~/.bashrc    # set conda path and environment variables 
[glogin01]$ conda config --set auto_activate_base false
[glogin01]$ which conda
/scratch/$USER/miniconda3/condabin/conda
[glogin01]$ conda --version
conda 25.1.1
```

## Cloning the Repository
to set up this repository on your scratch direcory.

* For Llama.cpp Build on A100
```
[glogin01]$ mkdir -p /scratch/$USER/llama.cpp
[glogin01]$ cd /scratch/$USER/llama.cpp
[glogin01]$ git clone https://github.com/ggml-org/llama.cpp.git llama.cpp.a100
[glogin01]$ cd llama.cpp.a100
```

* For Llama.cpp Build on H200
```
[glogin01]$ mkdir -p /scratch/$USER/llama.cpp
[glogin01]$ cd /scratch/$USER/llama.cpp
[glogin01]$ git clone https://github.com/ggml-org/llama.cpp.git llama.cpp.h200 
[glogin01]$ cd llama.cpp.h200
```

## Installing Llama.cpp
to build the llama.cpp locally on your scratch directory. Please refer to the [how to build llama.cpp locally here](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) for more details.
* For A100

Please be noted that the "glogin01" login node has A100 GPUs. 
```
[glogin01]$ pwd 
/scratch/$USER/llama.cpp/llama.cpp.a100
[glogin01]$ module load gcc/10.2.0 cuda/12.1 cmake/3.26.2
[glogin01]$ cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=1  -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="-mno-avx512vnni -mno-avx512bf16" -DCMAKE_CXX_FLAGS="-mno-avx512vnni -mno-avx512bf16"
[glogin01]$ cmake --build build --config Release -j 8
[glogin01]$ ls ./build/bin/
./                              llama-lookup*              llama-vdot*
../                             llama-lookup-create*       test-arg-parser*
llama-batched*                  llama-lookup-merge*        test-autorelease*
llama-batched-bench*            llama-lookup-stats*        test-backend-ops*
llama-bench*                    llama-minicpmv-cli*        test-barrier*
llama-cli*                      llama-parallel*            test-c*
llama-convert-llama2c-to-ggml*  llama-passkey*             test-chat*
llama-cvector-generator*        llama-perplexity*          test-chat-template*
llama-embedding*                llama-q8dot*               test-gguf*
llama-eval-callback*            llama-quantize*            test-grammar-integration*
llama-export-lora*              llama-quantize-stats*      test-grammar-parser*
llama-gbnf-validator*           llama-qwen2vl-cli*         test-json-schema-to-grammar*
llama-gen-docs*                 llama-retrieval*           test-llama-grammar*
llama-gguf*                     llama-run*                 test-log*
llama-gguf-hash*                llama-save-load-state*     test-model-load-cancel*
llama-gguf-split*               llama-server*              test-quantize-fns*
llama-gritlm*                   llama-simple*              test-quantize-perf*
llama-imatrix*                  llama-simple-chat*         test-rope*
llama-infill*                   llama-speculative*         test-sampling*
llama-llava-cli*                llama-speculative-simple*  test-tokenizer-0*
llama-llava-clip-quantize-cli*  llama-tokenize*            test-tokenizer-1-bpe*
llama-lookahead*                llama-tts*                 test-tokenizer-1-spm*
```
* For H200

You need to request a H200 node first to build the llama.cpp for H200.
```
[glogin01]$ salloc --partition=eme_h200nv_8 -J debug --nodes=1 --time=12:00:00 --gres=gpu:1  --comment pytorch
salloc: Granted job allocation 154173
salloc: Waiting for resource configuration
salloc: Nodes gpu48 are ready for job
[gpu48]$ module load gcc/10.2.0 cuda/12.1 cmake/3.26.2
[gpu48]$ cd /scratch/$USER/llama.cpp/llama.cpp.h200
[gpu48]$ cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_ENABLE_UNIFIED_MEMORY=1  -DBUILD_SHARED_LIBS=OFF -DCMAKE_C_FLAGS="-mno-avx512vnni -mno-avx512bf16" -DCMAKE_CXX_FLAGS="-mno-avx512vnni -mno-avx512bf16"
[gpu48]$ cmake --build build --config Release -j 8
[gpu48]$ ls ./build/bin
./                              llama-lookup*              llama-vdot*
../                             llama-lookup-create*       test-arg-parser*
llama-batched*                  llama-lookup-merge*        test-autorelease*
llama-batched-bench*            llama-lookup-stats*        test-backend-ops*
llama-bench*                    llama-minicpmv-cli*        test-barrier*
llama-cli*                      llama-parallel*            test-c*
llama-convert-llama2c-to-ggml*  llama-passkey*             test-chat*
llama-cvector-generator*        llama-perplexity*          test-chat-template*
llama-embedding*                llama-q8dot*               test-gguf*
llama-eval-callback*            llama-quantize*            test-grammar-integration*
llama-export-lora*              llama-quantize-stats*      test-grammar-parser*
llama-gbnf-validator*           llama-qwen2vl-cli*         test-json-schema-to-grammar*
llama-gen-docs*                 llama-retrieval*           test-llama-grammar*
llama-gguf*                     llama-run*                 test-log*
llama-gguf-hash*                llama-save-load-state*     test-model-load-cancel*
llama-gguf-split*               llama-server*              test-quantize-fns*
llama-gritlm*                   llama-simple*              test-quantize-perf*
llama-imatrix*                  llama-simple-chat*         test-rope*
llama-infill*                   llama-speculative*         test-sampling*
llama-llava-cli*                llama-speculative-simple*  test-tokenizer-0*
llama-llava-clip-quantize-cli*  llama-tokenize*            test-tokenizer-1-bpe*
llama-lookahead*                llama-tts*                 test-tokenizer-1-spm*
```

## Creating a Conda Virtual Environment
1. Create a conda virtual environment with a python version 3.11
```
[glogin01]$ conda create -n llama.cpp python=3.11
Retrieving notices: ...working... done
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3/envs/llama.cpp

  added / updated specs:
    - python=3.11
.
.
.
Proceed ([y]/n)? y    <========== type yes

Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate llama.cpp
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

2. Install PyTorch
```
[glogin01]$ module load gcc/10.2.0 cmake/3.26.2 cuda/12.1
[glogin01]$ conda activate llama.cpp
(llama.cpp) [glogin01]$ pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
Looking in indexes: https://download.pytorch.org/whl/cu121, https://pypi.ngc.nvidia.com
Collecting torch==2.5.0
  Downloading https://download.pytorch.org/whl/cu121/torch-2.5.0%2Bcu121-cp311-cp311-linux_x86_64.whl (780.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 780.5/780.5 MB 69.3 MB/s eta 0:00:00
.
.
.
Installing collected packages: mpmath, typing-extensions, sympy, pillow, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, networkx, MarkupSafe, fsspec, filelock, triton, nvidia-cusparse-cu12, nvidia-cudnn-cu12, jinja2, nvidia-cusolver-cu12, torch, torchvision, torchaudio
Successfully installed MarkupSafe-2.1.5 filelock-3.13.1 fsspec-2024.6.1 jinja2-3.1.4 mpmath-1.3.0 networkx-3.3 numpy-2.1.2 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.1.105 nvidia-nvtx-cu12-12.1.105 pillow-11.0.0 sympy-1.13.1 torch-2.5.0+cu121 torchaudio-2.5.0+cu121 torchvision-0.20.0+cu121 triton-3.1.0 typing-extensions-4.12.2

```
3. Install Open WebUI

You can install Open WebUI using pip. Please refer to the [Open WebUI documentation here](https://docs.openwebui.com/) for more details. 
```
(llama.cpp) [glogin01]$ pip install open-webui
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Collecting open-webui
  Downloading open_webui-0.5.16-py3-none-any.whl.metadata (19 kB)
.
.
.
Successfully installed Events-0.5 Mako-1.3.9 PyYAML-6.0.2 RTFDE-0.1.2 Shapely-2.0.7 XlsxWriter-3.2.2 aiocache-0.12.3 aiofiles-24.1.0 aiohappyeyeballs-2.4.6 aiohttp-3.11.11 aiosignal-1.3.2 alembic-1.14.0 annotated-types-0.7.0 anthropic-0.46.0 anyio-4.8.0 appdirs-1.4.4 apscheduler-3.10.4 argon2-cffi-23.1.0 argon2-cffi-bindings-21.2.0 ...
uvicorn-0.30.6 uvloop-0.21.0 validators-0.34.0 watchfiles-1.0.4 wcwidth-0.2.13 webencodings-0.5.1 websocket-client-1.8.0 websockets-15.0 werkzeug-3.1.3 wrapt-1.17.2 wsproto-1.2.0 xlrd-2.0.1 xmltodict-0.14.2 xxhash-3.5.0 yarl-1.18.3 youtube-transcript-api-0.6.3 zipp-3.21.0

```
4. Install Hugging Face dependencies for download a DeepSeek R1 Dynamic Quantization model
```
(llama.cpp) [glogin01]$ pip install huggingface_hub hf_transfer
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com
Requirement already satisfied: huggingface_hub in /scratch/qualis/miniconda3/envs/llama.cpp/lib/python3.11/site-packages (0.29.1)
Collecting hf_transfer
  Downloading hf_transfer-0.1.9-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.7 kB)
.
.
.
Installing collected packages: hf_transfer
Successfully installed hf_transfer-0.1.9
```

## Download the dynamic quantized version of DeepSeek-R1
For this test, we will download and use the 1.58-bit (131GB) version ("DeepSeek-R1-UD-IQ1_S"). Please refer to the UnslothAI's blog post: [Run DeepSeek R1
Dynamic 1.58-bit](https://unsloth.ai/blog/deepseekr1-dynamic) for more details about their dynamic quantized versions.
```
(llama.cpp) [glogin01]$ cat download_model.py
import os
from huggingface_hub import snapshot_download

local_dir = os.path.expandvars("/scratch/$USER/llama.cpp/DeepSeek-R1-GGUF")

snapshot_download(
    repo_id = "unsloth/DeepSeek-R1-GGUF",  # Specify the Hugging Face repo
    local_dir = local_dir,         # Model will download into this directory
    allow_patterns = ["*UD-IQ1_S*"],        # Only download the 1.58-bit version
)
```
```
(llama.cpp) [glogin01]$ python download_model.py
```
Once the download completes, you’ll find the model files in your directory (/scratch/$USER/llama.cpp):
```
DeepSeek-R1-GGUF/
├── DeepSeek-R1-UD-IQ1_S/
│   ├── DeepSeek-R1-UD-IQ1_S-00001-of-00003.gguf
│   ├── DeepSeek-R1-UD-IQ1_S-00002-of-00003.gguf
│   ├── DeepSeek-R1-UD-IQ1_S-00003-of-00003.gguf
```

## Running Open WebUI
This section describes **how to run the Open WebUI along with launching the llama.cpp server and Open WebUI server on a compute node.** The following Slurm script will start both servers and output a port forwarding command, which you can use to connect remotely.

### Slurm Script (llama_openwebui_run.sh)
```bash
#!/bin/bash
#SBATCH --comment=pytorch
##SBATCH --partition=amd_a100nv_8
#SBATCH --partition=eme_h200nv_8
#SBATCH --time=48:00:00
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-node=2
##SBATCH --gres=gpu:1
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
```

### Submitting the Slurm Script
- to launch both llama.cpp and Open WebUI server
```
(deepseek) [glogin01]$ sbatch llama_openwebui_run.sh
Submitted batch job XXXXXX
```
- to check if the servers are up and running
```
(deepseek) [glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            XXXXXX      ollama_g    $USER  RUNNING       0:02   2-00:00:00      1 gpu##
```
- to check the SSH tunneling information generated by the ollama_gradio_run.sh script 
```
(deepseek) [glogin01]$ cat port_forwarding_command
ssh -L localhost:8080:gpu50:8080 $USER@neuron.ksc.re.kr
```

### Connecting to the Open WebUI
- Once the job starts, open a a new SSH client (e.g., Putty, MobaXterm, PowerShell, Command Prompt, etc) on your local machine and run the port forwarding command displayed in port_forwarding_command:

<img width="851" alt="Image" src="https://github.com/user-attachments/assets/12ed1a81-3bef-4755-bae1-996118672a8a" />

- Then, open http://localhost:8080 in your browser to access the Open WebUI and create an admin account to connect the llama.cpp server running on the compute node.

<img width="1198" alt="Image" src="https://github.com/user-attachments/assets/d5f835bf-c344-4c9a-9526-d693e8d7ff9a" />

- Connect to the `llama.cpp` server as follows:
  - Go to **Settings** and **Admin Settings** by clicking the orange admin account at the bottom left in Open WebUI.
  - Navigate to **Connections > OpenAI Connections**.
  - Add the following details for the new connection:
    - **URL:** `http://127.0.0.1:10000/v1`
    - **API Key:** `none`

<img width="558" alt="Image" src="https://github.com/user-attachments/assets/4916b9c0-2c10-41b8-bd61-80debb6d1e12" />

- Once the connection is saved, you can start to use Open WebUI’s chat interface to interact with the DeepSeek-R1 Dynamic 1.58-bit model.

![Image](https://github.com/user-attachments/assets/e2083708-d78e-4258-ad44-ed0faf01df21)


## Reference
* [Run DeepSeek R1 Dynamic 1.58-bit with Llama.cpp](https://docs.openwebui.com/tutorials/integrations/deepseekr1-dynamic/)
