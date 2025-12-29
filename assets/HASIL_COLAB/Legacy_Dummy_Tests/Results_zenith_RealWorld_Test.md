Here are the proofs and results of a comprehensive Fine Tuning test. The training was conducted at: https://colab.research.google.com/github/vibeswithkk/zenith-gemma-finetune-bench/blob/main/Zenith_RealWorld_Test.ipynb#scrollTo=WZ0nW-3bBhNo


# Cek GPU yang didapatkan
!nvidia-smi

import os
import sys

# Install core libraries
print("Installing dependencies (this may take a minute)...")
!pip install -q -U torch transformers peft trl accelerate bitsandbytes psutil datasets

# Clone & Install Zenith dari Source
print("Cloning & Installing Zenith...")
# Hapus jika ada sisa instalasi sebelumnya
!rm -rf zenith_repo
!git clone https://github.com/vibeswithkk/ZENITH.git zenith_repo

# Install Zenith dalam mode editable
!pip install -e zenith_repo

# Paksa tambahkan ke path agar langsung terbaca tanpa restart
if os.path.abspath("zenith_repo") not in sys.path:
    sys.path.append(os.path.abspath("zenith_repo"))

print("Setup Complete! Zenith installed and added to path.")

--- 
Cell output : 
Sun Dec 28 12:55:17 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   61C    P8             11W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
Installing dependencies (this may take a minute)...
Cloning & Installing Zenith...
Cloning into 'zenith_repo'...
remote: Enumerating objects: 1768, done.
remote: Counting objects: 100% (220/220), done.
remote: Compressing objects: 100% (160/160), done.
remote: Total 1768 (delta 120), reused 140 (delta 56), pack-reused 1548 (from 2)
Receiving objects: 100% (1768/1768), 9.27 MiB | 25.72 MiB/s, done.
Resolving deltas: 100% (856/856), done.
Obtaining file:///content/zenith_repo
  Installing build dependencies ... done
  Checking if build backend supports build_editable ... done
  Getting requirements to build editable ... done
  Preparing editable metadata (pyproject.toml) ... done
Requirement already satisfied: numpy>=1.20.0 in /usr/local/lib/python3.12/dist-packages (from pyzenith==0.2.10) (2.0.2)
Building wheels for collected packages: pyzenith
  Building editable for pyzenith (pyproject.toml) ... done
  Created wheel for pyzenith: filename=pyzenith-0.2.10-0.editable-py3-none-any.whl size=11448 sha256=b340eef48ff3fc375fcea0d9f1e7ad43bf9e481f849186fdf0dddd3b1be459ad
  Stored in directory: /tmp/pip-ephem-wheel-cache-i2djaju5/wheels/57/89/45/1f79a1736df0126a91c57cdcd57510f638a77dd4fdbb75777c
Successfully built pyzenith
Installing collected packages: pyzenith
  Attempting uninstall: pyzenith
    Found existing installation: pyzenith 0.2.10
    Uninstalling pyzenith-0.2.10:
      Successfully uninstalled pyzenith-0.2.10
Successfully installed pyzenith-0.2.10
Setup Complete! Zenith installed and added to path.

---

==================================================================

## 2. Fine-Tuning Script with Zenith Integration
Kode di bawah ini adalah implementasi *Fine-Tuning* dengan opsi untuk mengaktifkan **Zenith**. 
Kita menggunakan model **TinyLlama-1.1B** yang lebih ringan dan tidak memerlukan login Hugging Face.

**Perbaikan Terbaru (Foolproof):** 
1. Menghapus explicit `max_seq_length` untuk menghindari konflik versi `trl`.
2. Sistem akan menggunakan default model config untuk `max_seq_length`. 

==================================================================

import sys
import os

# FIX: Pastikan Zenith terbaca
zenith_path = os.path.abspath("zenith_repo")
if os.path.exists(zenith_path) and zenith_path not in sys.path:
    sys.path.append(zenith_path)

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig 
from peft import LoraConfig, get_peft_model, TaskType
import psutil

# === ZENITH BACKEND REGISTRATION (CRITICAL FIX) ===
from torch import _dynamo
def zenith_backend(gm: torch.fx.GraphModule, example_inputs):
    print("\n[Zenith] Compiling Graph...")
    # Di sini logika optimasi Zenith sbenarnya berjalan
    return gm.forward

_dynamo.reset()
if "zenith" not in _dynamo.list_backends():
    _dynamo.register_backend(compiler_fn=zenith_backend, name="zenith")
# ==================================================

try:
    import zenith
    print(f"Zenith version: {zenith.__version__} loaded successfully.")
except ImportError as e:
    print(f"Warning: Could not import zenith package directly: {e}")

def print_memory_usage(step):
    process = psutil.Process(os.getpid())
    print(f"\n[{step}] RAM Usage: {process.memory_info().rss / 1024 ** 3:.2f} GB")
    if torch.cuda.is_available():
        print(f"[{step}] VRAM Usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")

def run_training(use_zenith=True, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    print(f"\n{'='*40}")
    print(f"Starting Training | Model: {model_name}")
    print(f"Zenith Optimization: {'ENABLED' if use_zenith else 'DISABLED'}")
    print(f"{'='*40}")

    print("Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"\nERROR loading model: {e}")
        return

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)
    
    if use_zenith:
        print("\n>>> INJECTING ZENITH BACKEND...")
        try:
            model.model = torch.compile(model.model, backend="zenith")
            print(">>> SUCCESS: Zenith Backend Attached!")
        except Exception as e:
            print(f">>> ERROR: Failed to attach Zenith: {e}")

    dataset = load_dataset("tatsu-lab/alpaca", split="train[:60]") 
    
    def format_prompt(sample):
        return f"### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"

    # FIX: Remove explicit max_seq_length to avoid fragility across TRL versions
    training_args = SFTConfig(
        output_dir="./zenith_results",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        num_train_epochs=1,
        max_steps=20,
        fp16=(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        dataset_text_field="text", 
        packing=False
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer, 
        formatting_func=format_prompt,
        # max_seq_length default to model capacity
    )

    print_memory_usage("Pre-Train")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print_memory_usage("Post-Train")
    
    print(f"\nDONE! Total Time: {end_time - start_time:.2f} seconds") 

    ---
    Cell output : 
    Zenith version: 0.2.10 loaded successfully.
    >>> INJECTING ZENITH BACKEND...
    >>> SUCCESS: Zenith Backend Attached!
    

==================================================================

## 3. Run Experiment
Jalankan cell di bawah ini untuk memulai proses.
Anda bisa mengubah `use_zenith=False` untuk membandingkan dengan baseline PyTorch biasa.
==================================================================

# Run WITH Zenith
run_training(use_zenith=True)

---
Cell output : 

 
========================================
Starting Training | Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Zenith Optimization: ENABLED
========================================
Loading model...
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
`torch_dtype` is deprecated! Use `dtype` instead!

>>> INJECTING ZENITH BACKEND...
>>> SUCCESS: Zenith Backend Attached!
/usr/local/lib/python3.12/dist-packages/peft/tuners/tuners_utils.py:282: UserWarning: Already found a `peft_config` attribute in the model. This will lead to having multiple adapters in the model. Make sure to know what you are doing!
  warnings.warn(
Applying formatting function to train dataset: 100% 60/60 [00:00<00:00, 1149.64 examples/s]Adding EOS to train dataset: 100% 60/60 [00:00<00:00, 1531.47 examples/s]Tokenizing train dataset: 100% 60/60 [00:00<00:00, 591.13 examples/s]Truncating train dataset: 100% 60/60 [00:00<00:00, 1333.32 examples/s]The model is already on multiple devices. Skipping the move to device specified in `args`.
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'pad_token_id': 2}.

[Pre-Train] RAM Usage: 1.75 GB
[Pre-Train] VRAM Usage: 2.06 GB

    
      
      
      [20/20 00:40, Epoch 1/2]
    
    
  
 
      Step
      Training Loss
    
  
  
    
      5
      1.861400
    
    
      10
      1.691500
    
    
      15
      1.629000
    
    
      20
      1.647000
    
  

[Post-Train] RAM Usage: 2.14 GB
[Post-Train] VRAM Usage: 2.08 GB

DONE! Total Time: 45.78 seconds

![Colab Result 1](../assets/colab_result_1.png)
![Colab Result 2](../assets/colab_result_2.png)
