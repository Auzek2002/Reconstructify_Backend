# Reconstructify 🔍🧬

**Reconstructify** is a deep learning-based tool that leverages **diffusion models** to reconstruct **partial fingerprints**. It then compares the regenerated fingerprints with a database of complete fingerprints to assist in identification or verification tasks.

## 🧠 Key Features

- ✨ Uses **state-of-the-art diffusion models** for fingerprint reconstruction.
- 🔎 Compares reconstructed prints with a database of full fingerprints.
- 📊 Provides similarity scores or match results for each comparison.
- 🖼️ Visualizes generated fingerprint and closest database match.

## 🛠️ How It Works

1. **Input**: Upload a partial fingerprint image.
2. **Reconstruction**: The model reconstructs the missing or damaged parts using a trained diffusion model.
3. **Comparison**: The output is matched against a database of complete fingerprints.
4. **Output**: Results include the closest matching fingerprint(s) and similarity scores.


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- npm

### Installation & Running

```bash
git clone https://github.com/Auzek2002/Reconstructify.git
cd Reconstructify
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Model Notebook:
[Open Notebook in Colab](https://colab.research.google.com/drive/1Ya6O2ive9Ld5t3PlwE-d1pAAZojFrOSn)

## Project Demo:

https://github.com/user-attachments/assets/44c66697-77b1-4162-b7d3-0a840bed4856


