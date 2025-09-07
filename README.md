# Real-Time Emotion Detection on GPU (FER+ ONNX) — College Project

This project gives you a **clean, error-proof baseline** for your college requirement:

**Concept:** Parallelize inference of a CNN-based emotion detection model on faces using **CUDA**  
**Outcome:** Demonstrate **FPS improvement** over a **CPU baseline**

You will run the same pipeline twice:
1) **CPU baseline** on your laptop (no NVIDIA GPU needed).  
2) **GPU run** on Google Colab (free NVIDIA GPU) with CUDA via **ONNX Runtime (CUDA EP)**.

We use:
- **OpenCV DNN** SSD face detector (ResNet-10) to crop faces.
- **FER+ pre-trained CNN** (`emotion-ferplus-8.onnx`) for emotion classification (8 classes).
- **ONNX Runtime** for inference (CPU or CUDA).

> Tip: The overall FPS includes face detection (CPU). The **classification step** gets the big speed-up on GPU. Use the `--bench-classifier-only` flag to measure classifier speed precisely.

## Folder layout
```
emotion_gpu_project/
  models/                # model files will be auto-downloaded here at first run
  src/
    common.py            # utilities (downloader, drawing, softmax, FPS meter)
    models.py            # model URLs + loaders (face detector + FER+ ONNX)
    run_realtime.py      # main webcam/video app (CPU/GPU switchable)
    benchmark_video.py   # quick FPS benchmark on a saved video (optional)
  requirements_cpu.txt
  requirements_colab.txt
  colab_gpu_instructions.txt
  README.md
```

## Quick Start (CPU baseline on your laptop — Windows PowerShell)

1. Create & activate a virtual environment:
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_cpu.txt
```

2. Run on your **webcam** (press **q** to quit):
```powershell
python -m src.run_realtime --source 0 --show 1
```

3. Or run on a **video file**:
```powershell
python -m src.run_realtime --source "demo.mp4" --show 1
```

4. Save an **annotated output video** and **CSV FPS log**:
```powershell
python -m src.run_realtime --source 0 --show 1 --save out.avi --log_fps fps_cpu.csv
```

> If `py` isn't found, replace with `python`.
> If your webcam index 0 doesn't work, try `--source 1`.

## GPU Run on **Google Colab** (CUDA speed-up)

Open Colab, **Runtime → Change runtime type → GPU**.  
Then copy-paste the commands from `colab_gpu_instructions.txt` into a Colab cell, run, and then:
```bash
python -m src.run_realtime --source 0 --show 1 --providers CUDAExecutionProvider --log_fps fps_gpu.csv
```
You can also test with a short sample video (upload it to Colab first):
```bash
python -m src.run_realtime --source sample.mp4 --providers CUDAExecutionProvider --show 1
```

## What to report
- **CPU vs GPU FPS** (from `fps_cpu.csv` and `fps_gpu.csv` or console print).
- Add `--bench-classifier-only 1` to isolate classifier speed.
- Optional: Increase parallelism with `--batch 4` to batch multiple faces per frame.

## Notes
- Face detector runs on **CPU** (OpenCV pip wheel does not ship with CUDA). Classification runs on **GPU** in Colab via ONNX Runtime. This satisfies *CUDA parallelization* for your CNN classifier.
- Emotions: `['neutral','happiness','surprise','sadness','anger','disgust','fear','contempt']`.

## References
- FER+ labels and ordering (Microsoft): https://github.com/microsoft/FERPlus  
- ONNX FER+ model (emotion-ferplus-8.onnx): https://github.com/onnx/models/tree/main/vision/body_analysis/emotion_ferplus  
- OpenCV SSD face detector (ResNet-10) model + preprocessing (mean/scale/300x300): see OpenCV `models.yml`.  
