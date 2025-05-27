# AuraSR Gradio Interface

This repository provides a Gradio-based interface for the [AuraSR](https://github.com/fal-ai/aura-sr) super-resolution model, enabling easy 4x upscaling of images. Drop an image, click "Upscale," and download the high-resolution result (e.g., 1024x1024 to 4096x4096). Unlike the original AuraSR, which expects 256x256 inputs, this interface supports arbitrary input sizes, making it ideal for high-resolution upscaling. Most modern GPUs can handle 4k (4096x4096) outputs without issues, though performance depends on your hardware.

## Features
- **Drag-and-Drop Interface**: Upload any image via Gradio's web interface.
- **4x Upscaling**: Upscales images by 4x (e.g., 1024x1024 to 4096x4096) using AuraSR's GAN-based model.
- **Downloadable Results**: Save upscaled images directly from the interface.
- **Windows-Compatible**: Setup instructions tailored for Windows users.
- **GPU Support**: Leverages CUDA-accelerated PyTorch for fast processing.

## Citation
This project builds on the [AuraSR repository](https://github.com/fal-ai/aura-sr) by fal-ai, which provides the core super-resolution model. The model weights and configuration are automatically downloaded from [fal-ai/AuraSR](https://huggingface.co/fal-ai/AuraSR) on first run.

## Requirements
- Windows 10 or 11
- Python 3.10 or 3.11
- A CUDA-compatible GPU (NVIDIA) with CUDA 12.6 installed (most modern GPUs can handle 4k upscaling)
- At least 8GB VRAM for 4k outputs (16GB recommended for larger images)

## Setup Instructions (Windows)

1. **Clone the Repository**:
   Open Command Prompt or PowerShell and run:
   ```bash
   git clone https://huggingface.co/<your-username>/<your-repo-name>](https://github.com/DragonDiffusionbyBoyo/Boyo-Aura-SR
   cd Boyo-Aura-SR
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   The `requirements.txt` includes CUDA-accelerated PyTorch for optimal performance. Run:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` specifies:
   ```
Pillow
requests
gradio
safetensors
numpy
einops
huggingface_hub
safetensors
gradio
torch==2.5.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
torchvision==0.20.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
torchaudio==2.5.0+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
   ```

4. **Verify CUDA**:
   Ensure CUDA 12.6 is installed (check with `nvcc --version`). If not, download from [NVIDIA's CUDA Toolkit](https://developer.nvidia.com/cuda-12-6-0-download-archive). Confirm GPU support:
   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```
   Expected output: `2.5.0+cu126 True`

5. **Run the Gradio Interface**:
   ```bash
   python aura_sr_gradio.py
   ```
   A browser window will open with the Gradio interface.

## Usage
1. Open the Gradio interface in your browser (typically `http://localhost:7860`).
2. Drag and drop an image (e.g., 1024x1024 PNG or JPEG).
3. Click "Upscale" to process the image (outputs 4096x4096 for a 1024x1024 input).
4. View the upscaled image and click "Download Upscaled Image" to save it.

## Notes
- **Input Sizes**: Unlike the original AuraSR, this interface supports arbitrary input sizes, upscaling them by 4x (e.g., 512x768 → 2048x3072, 1024x1024 → 4096x4096).
- **GPU Performance**: Most GPUs (e.g., NVIDIA RTX 2060 or better) can handle 4k upscaling without overheating. For larger images (e.g., 8k+), ensure sufficient VRAM (16GB+ recommended).
- **Image Quality**: Use high-quality inputs (PNG or lossless formats) for best results, as AuraSR is sensitive to compression artifacts.
- **Troubleshooting**: If the output size matches the input, check the console for errors. Verify input/output sizes with debug logs in `aura_sr_gradio.py`.

## License
This project is licensed under the same terms as the original [AuraSR repository](https://github.com/fal-ai/aura-sr) (Apache 2.0). See the original repository for details.

## Acknowledgments
- Thanks to [fal-ai](https://github.com/fal-ai) for developing AuraSR.
- Gradio interface inspired by community feedback and adapted for flexible input sizes.
