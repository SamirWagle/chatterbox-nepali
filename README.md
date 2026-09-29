# 🇳🇵 Chatterbox-TTS (Nepali Edition)

Experimental Nepali text-to-speech adaptation based on the **Chatterbox-Multilingual-500M** architecture. This repository contains custom training logic, Devanagari fixes, and voice-cloning experiments. Formal perceptual-quality evaluation remains open work.

![Nepali TTS Preview](./Chatterbox-Multilingual.png)

## 🚀 Key Features
* **Full Devanagari Support**: Patched tokenizer and alignment analyzer to handle complex Nepali character clusters without cutting off.
* **Voice Cloning Workflow**: Zero-shot cloning experiments using a 5-10 second reference clip.
* **Optimized for Mac**: Pre-configured for **Apple Silicon (MPS)** acceleration and memory-efficient training on M2/M3 chips.
* **Clean Inference**: Dedicated Gradio UI and test scripts for rapid experimentation.

## 📦 Model Checkpoints
To ensure repository performance, large training checkpoints are hosted on Hugging Face:
- **Model Link**: [https://huggingface.co/waglesameer5/chatterbox-nepali](https://huggingface.co/waglesameer5/chatterbox-nepali)

| File | Purpose | Recommendation |
| :--- | :--- | :--- |
| `t3_nepali_epoch_20.pt` | **Base Nepali Checkpoint** | Use for **quick testing** or as a starting point to **train further**. |

> [!TIP]
> The provided checkpoint is an experimental starting point. More epochs do not guarantee better perceptual quality; validate every checkpoint with held-out text and structured listening tests.

*Place this file in the root folder of this repository after downloading.*

---

## ⚙️ Installation
Follow these steps to set up the environment and download the codebase:

```bash
# 1. Create and activate a dedicated conda environment
conda create -n chatterbox_ne python=3.11 -y
conda activate chatterbox_ne

# 2. Clone the repository
git clone https://github.com/SamirWagle/chatterbox-nepali.git
cd chatterbox-nepali

# 3. Install dependencies in editable mode
pip install -e .
```

---

## 🎙️ Inference & Implementation

### 🛡️ Quick Testing
To generate Nepali speech correctly, you **must** use the provided test scripts. Standard library imports from Hugging Face will not support Devanagari without these specific patches.

#### 🎧 Hear the Results
**[Listen to the Nepali Rainbow Benchmark Output](./nepali_test_output.wav)**

#### Benchmark (M2 Max 64GB)
- **Input**: Long paragraph (~45 words)
- **Reference Audio**: 10 seconds (samples/achyut_ref_10s.wav)
- **Generation Time (MPS)**: **~115 seconds**
- **Real-time Factor**: ~0.35x

#### Run the test command:
```bash
# Make sure your environment is active
conda activate chatterbox_ne
export PYTHONPATH=src

python3 test_nepali.py \
  --checkpoint "t3_nepali_epoch_20.pt" \
  --ref_audio "samples/achyut_ref_10s.wav" \
  --text "इन्द्रेणी वा इन्द्रधनुष प्रकाश र रंगबाट उत्पन्न भएको यस्तो घटना हो जसमा रंगीन प्रकाशको एउटा अर्धवृत आकाशमा देखिन्छ। जब सूर्यको प्रकाश पृथ्वीको वायुमण्डलमा भएको पानीको थोपा माथि पर्छ, पानीको थोपाले प्रकाशलाई परावर्तन, आवर्तन र डिस्पर्सन गर्दछ।" \
  --output "nepali_test_output.wav"
```

> [!NOTE]
> The file **`nepali_test_output.wav`** in the root of this repository was generated using the command above. For testing emotional expressiveness and paralinguistic control, we recommend using the **Gradio Interface**. Note: The current Gradio script is optimized for the final `.safetensors` model; you can modify `gradio_nepali.py` if you wish to host a raw `.pt` checkpoint instead.

### 🏮 Web UI (Gradio)
Launch a graphical interface to test voices instantly. It will automatically detect and load your local Nepali weights:
```bash
conda activate chatterbox_ne
export PYTHONPATH=src
python3 gradio_nepali.py
```

---

## 🏋️ Training / Dataset Format
If you wish to fine-tune the model further or use your own voice data, ensure your dataset follows the standard format:

### 1. File Structure
```text
data/nepali/
├── metadata.csv
└── wavs/
    ├── voice_01.wav
    ├── voice_02.wav
    └── ...
```

### 2. `metadata.csv` (Pipe-separated)
The file should **not** have a header. Use the `filename|[ne]text` format:
```csv
voice_01|[ne]नमस्ते संसार यो मेरो आवाज हो
voice_02|[ne]नेपाली भाषा धेरै मीठो छ
```

### 3. Audio Requirements
*   **Format**: Mono WAV (24,000 Hz or 48,000 Hz recommended).
*   **Duration**: 2 to 10 seconds per clip.

### 4. Resume Training
```bash
conda activate chatterbox_ne
export PYTHONPATH=src

python3 src/chatterbox/train_nepali.py \
  --manifest data/nepali/metadata.csv \
  --device mps \
  --batch_size 4 \
  --accum_steps 4 \
  --epochs 50 \
  --save_every 5 \
  --resume_t3_weights "t3_nepali_epoch_20.pt"
```

---

## 🎯 Post-Training (Safetensors Generation)
Once your training reaches the final epoch (e.g. 50), the script will automatically consolidate your efforts and generate a high-performance **`t3_mtl_nepali_final.safetensors`** file.

### Using the exported Safetensors checkpoint
Safetensors avoids pickle-based deserialization. Treat the exported checkpoint as an evaluation artifact until quality, latency, provenance, and deployment behavior are validated for your use case.

```bash
# Use the same test script but point to your final safetensors
python3 test_nepali.py \
  --checkpoint "t3_mtl_nepali_final.safetensors" \
  --ref_audio "your_reference.wav" \
  --text "तपाईंको नयाँ नेपाली एआई तयार छ।" \
  --output "production_output.wav"
```

---

## 🛠️ Nepali adaptation changes
This adaptation includes changes required by its current Devanagari workflow:
* **Causal Shift Fix**: Fixed the next-token prediction loss in `t3.py`.
* **Tokenizer Logic**: Prevented double-prepending of `[ne]` tags.
* **Alignment Safety**: Increased repetition tolerance in `alignment_stream_analyzer.py` to stop early audio cutoffs on long Nepali vowels.

## 📄 License & Credits
* Original architecture by **Resemble AI**.
* Fine-tuning and Nepali adaptation by **Samir Wagle**.
* Reference Audio in samples by **Achyut Ghimire (Bulbul)**.
* Distributed under the **MIT License**.
