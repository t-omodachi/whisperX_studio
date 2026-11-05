# WhisperX Studio

A local web app that transcribes video and audio files and labels who is speaking. Everything runs on your own computer: your files and transcripts are never uploaded anywhere.

## What you need

- Windows 10 or 11
- Python 3.11
- An NVIDIA GPU with at least 6 GB of memory is strongly recommended. It also runs without one, but much more slowly.
- About 6 GB of free disk space (packages plus speech models)

## Setup (one time)

Open PowerShell in this folder and run these one at a time.

1. Create a Python environment:

   ```bash
   py -3.11 -m venv .venv
   ```

2. Install PyTorch with GPU support:

   ```bash
   .venv\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
   ```

   No NVIDIA GPU? Use `https://download.pytorch.org/whl/cpu` at the end instead.

3. Install everything else:

   ```bash
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

If you use [uv](https://docs.astral.sh/uv/), the same steps are `uv venv --python 3.11`, `uv pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126`, then `uv pip install -r requirements.txt`.

FFmpeg doesn't need a separate install. The app uses a copy that comes with the Python packages.

## Running it

Double-click `Start-WhisperX-Studio.bat`. It starts the app and opens http://127.0.0.1:8765 in your browser. Keep the black window open while you use it, and close it to stop the app.

**First run:** the speech models (about 3.5 GB) download from Hugging Face the first time you transcribe something, so that run needs internet and takes a while. After that they are stored in the `models` folder and transcription works offline.

## Using it

1. Upload a video or audio file, or paste the full path of a file on your computer. Pasting a path is faster for big files because nothing gets copied.
2. Choose the language, or leave it on auto-detect.
3. Optionally set how many speakers there are. Otherwise the app guesses.
4. Start the transcription and wait for the progress bar to finish.

Leave the **Hugging Face token** field empty. Speaker detection works fully offline without it. A token switches to a different speaker model that has to be downloaded from the internet.

## Privacy

- The app only accepts connections from your own computer, not from your network or the internet.
- Uploaded files are kept in `uploads/` and transcripts in `outputs/`. Both stay on your computer, and `.gitignore` keeps them out of git.
- Delete those two folders' contents when you no longer need them.

## Troubleshooting

- **"An Application Control policy has blocked this file":** Windows Smart App Control blocked a package file while it checked it. Run the app again. It usually allows the file on the next try.
- **Runs slowly / "CPU Only" shown in the app:** PyTorch was installed without GPU support. Redo step 2 with the `cu126` link.
- **Out-of-memory error on the GPU:** close other programs that use the GPU and try again.
