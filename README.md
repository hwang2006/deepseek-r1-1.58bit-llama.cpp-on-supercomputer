# Running Deepseek-R1 Dynamic 1.58-bit with Llama.cpp on a Supercomputer

DeepSeek-R1, the recently released AI reasoning model from the Chinese AI startup DeepSeek, has gained significant attention for its performance, comparable to leading models like OpenAI's o1 reasoning model. It is open-source and free to use, allowing users to download, modify, and run it for their own purposes.

This repository demonstrates how to run and test DeepSeek-R1 in its dynamic 1.58-bit quantized form using Llama.cpp on a SLURM-managed supercomputer. Thanks to advanced quantization techniques, the full 671B parameter model is compressed to just 131GB, making it significantly more accessible. You can now efficiently run it on a supercomputer with 1 or 2 A100 or H200 GPUs allocated through your account, with 2 GPUs recommended for optimal performance. This removes the need for extremely large GPU configurations.

Llama.cpp provides an efficient framework for running large-scale AI models on CPUs and GPUs with optimized inference. This guide walks you through the steps needed to set up, deploy, and interact with DeepSeek-R1 in its quantized form on a high-performance computing (HPC) environment.

**Features:**

*   **Dynamic 1.58-bit quantization**: Reduces memory usage while maintaining model performance.
*   **Supercomputer compatibility**: Run DeepSeek-R1 efficiently on SLURM-managed HPC clusters.
*   **llama.cpp integration**: Utilize the optimized inference engine for fast execution.
*   **Supports 1-2 GPUs**: Can be run on a supercomputer with 1 or 2 A100 or H200 GPUs using individual user accounts.

**Note that 

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

## Building Llama.cpp
to install the llama.cpp locally on your scratch directory. 
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

You need to request a H200 node first to build llama.cpp for H200.
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
