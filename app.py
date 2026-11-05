import os
import sys
import gc
import json
import time
import uuid
import shutil
import asyncio
import threading
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

# ---------------------------------------------------------------------------
# 1. Windows CUDA DLL & FFmpeg Auto-Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VENV_SCRIPTS = BASE_DIR / ".venv" / "Scripts"
SITE_PACKAGES = BASE_DIR / ".venv" / "Lib" / "site-packages"


def setup_environment():
    """Ensure FFmpeg and NVIDIA CUDA DLLs are on PATH for Windows."""
    paths_to_add = [str(VENV_SCRIPTS)]

    # Locate NVIDIA CUDA DLL directories installed inside site-packages
    nvidia_dir = SITE_PACKAGES / "nvidia"
    if nvidia_dir.exists():
        for bin_dir in nvidia_dir.glob("*/bin"):
            if bin_dir.is_dir():
                paths_to_add.append(str(bin_dir))
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(str(bin_dir))
                    except Exception:
                        pass

    torch_lib = SITE_PACKAGES / "torch" / "lib"
    if torch_lib.exists():
        paths_to_add.append(str(torch_lib))
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(torch_lib))
            except Exception:
                pass

    # Ensure ffmpeg.exe exists in .venv/Scripts from imageio_ffmpeg if not on system PATH
    ffmpeg_target = VENV_SCRIPTS / "ffmpeg.exe"
    if not ffmpeg_target.exists() and not shutil.which("ffmpeg"):
        try:
            import imageio_ffmpeg
            src_ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
            if src_ffmpeg.exists():
                VENV_SCRIPTS.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_ffmpeg, ffmpeg_target)
        except Exception as e:
            print(f"[Warning] Could not auto-copy ffmpeg: {e}")

    os.environ["PATH"] = os.pathsep.join(paths_to_add) + os.pathsep + os.environ.get("PATH", "")
    # Avoid HuggingFace symlink privilege errors on Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


setup_environment()

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# Fix PyTorch 2.6+ weights_only=True default breaking pyannote/whisperx checkpoints
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

app = FastAPI(title="WhisperX Studio - Local Video Transcriber & Speaker Diarizer")

UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = BASE_DIR / "static"

for d in (UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR, STATIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Global job state store
JOBS: Dict[str, Dict[str, Any]] = {}
GPU_LOCK = threading.Lock()


def free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_system_info() -> Dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Only"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        if cuda_available
        else 0
    )
    return {
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "ffmpeg_available": bool(shutil.which("ffmpeg")),
    }


def extract_wav_16k(input_media: Path, output_wav: Path):
    """Extract 16kHz mono PCM WAV from any video or audio file using FFmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_media),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_wav),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not output_wav.exists():
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{proc.stderr[-600:]}")


# ---------------------------------------------------------------------------
# 2. Neural Speaker Diarization (Zero-Token ECAPA-TDNN + Optional Pyannote)
# ---------------------------------------------------------------------------
_ECAPA_CLASSIFIER = None


def load_ecapa_encoder(device: str):
    """Load SpeechBrain ECAPA-TDNN speaker embedding model without Windows symlink issues."""
    global _ECAPA_CLASSIFIER
    if _ECAPA_CLASSIFIER is not None:
        return _ECAPA_CLASSIFIER

    # Patch torchaudio compatibility if needed for newer torchaudio versions
    import torchaudio
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]

    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    save_dir = MODELS_DIR / "spkrec-ecapa-voxceleb"
    _ECAPA_CLASSIFIER = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(save_dir),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )
    return _ECAPA_CLASSIFIER


def split_segments_Into_phrases(segments: List[Dict[str, Any]], max_gap: float = 0.45, max_dur: float = 6.0) -> List[Dict[str, Any]]:
    """
    Use WhisperX word-level timestamps to split coarse segments at natural pauses
    or punctuation boundaries so speaker switches inside a segment are cleanly separated.
    """
    phrases: List[Dict[str, Any]] = []
    for seg in segments:
        words = seg.get("words", [])
        # Filter words that have valid start & end timestamps
        valid_words = [w for w in words if "start" in w and "end" in w]
        if not valid_words:
            phrases.append({
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", "").strip(),
                "words": words,
            })
            continue

        current_words = [valid_words[0]]
        for w in valid_words[1:]:
            prev_w = current_words[-1]
            gap = float(w["start"]) - float(prev_w["end"])
            cur_dur = float(prev_w["end"]) - float(current_words[0]["start"])
            prev_text = prev_w.get("word", "").strip()
            ends_sentence = prev_text.endswith((".", "?", "!"))

            if gap >= max_gap or (ends_sentence and gap >= 0.18) or (ends_sentence and cur_dur >= max_dur):
                phrases.append({
                    "start": float(current_words[0]["start"]),
                    "end": float(current_words[-1]["end"]),
                    "text": " ".join(x.get("word", "").strip() for x in current_words).strip(),
                    "words": current_words,
                })
                current_words = [w]
            else:
                current_words.append(w)

        if current_words:
            phrases.append({
                "start": float(current_words[0]["start"]),
                "end": float(current_words[-1]["end"]),
                "text": " ".join(x.get("word", "").strip() for x in current_words).strip(),
                "words": current_words,
            })

    return phrases


def diarize_with_ecapa(
    wav_path: Path,
    segments: List[Dict[str, Any]],
    device: str,
    num_speakers: Optional[int] = None,
    min_speakers: int = 1,
    max_speakers: int = 8,
) -> List[Dict[str, Any]]:
    """
    High-accuracy local speaker diarization using WhisperX word-aligned phrases
    + SpeechBrain ECAPA-TDNN speaker embeddings + Agglomerative Cosine Clustering.
    Works 100% locally with zero HuggingFace tokens required.
    """
    phrases = split_segments_Into_phrases(segments)
    if not phrases:
        return []

    audio_np, sr = sf.read(str(wav_path), dtype="float32")
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=1)
    total_samples = len(audio_np)

    classifier = load_ecapa_encoder(device)

    embeddings = []
    valid_indices = []

    # Extract 192-D ECAPA-TDNN embedding for each phrase
    # For very short phrases (< 0.8s), pad slightly with surrounding context from the phrase window
    for idx, p in enumerate(phrases):
        start_s = max(0.0, p["start"])
        end_s = min(total_samples / sr, p["end"])
        dur = end_s - start_s
        if dur < 0.15:
            continue

        # Ensure at least 0.9s window centered on the phrase for reliable voiceprint extraction
        if dur < 0.9:
            pad = (0.9 - dur) / 2.0
            s_idx = max(0, int((start_s - pad) * sr))
            e_idx = min(total_samples, int((end_s + pad) * sr))
        else:
            s_idx = max(0, int(start_s * sr))
            e_idx = min(total_samples, int(end_s * sr))

        chunk = audio_np[s_idx:e_idx]
        if len(chunk) < int(0.2 * sr):
            continue

        wav_tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(wav_tensor)
            emb_vec = emb.squeeze().cpu().numpy()
            embeddings.append(emb_vec)
            valid_indices.append(idx)

    if len(embeddings) <= 1:
        for p in phrases:
            p["speaker"] = "Speaker 1"
        return merge_consecutive_speaker_segments(phrases)

    X = normalize(np.vstack(embeddings), axis=1)

    # Determine optimal number of speakers if not fixed
    if num_speakers is not None and num_speakers >= 1:
        best_k = min(num_speakers, len(X))
    else:
        min_k = max(1, min_speakers)
        max_k = min(max_speakers, len(X) - 1, 10)
        if max_k < 2:
            best_k = 1
        else:
            # Check if all segments belong to a single speaker via pairwise cosine distance
            sim_matrix = np.dot(X, X.T)
            upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            median_sim = float(np.median(upper_tri)) if len(upper_tri) > 0 else 1.0

            if min_k == 1 and median_sim > 0.68 and float(np.percentile(upper_tri, 10)) > 0.52:
                best_k = 1
            else:
                best_k = max(2, min_k)
                best_score = -1.0
                for k in range(max(2, min_k), max_k + 1):
                    clustering = AgglomerativeClustering(
                        n_clusters=k,
                        metric="cosine",
                        linkage="average",
                    )
                    labels = clustering.fit_predict(X)
                    # Require at least 2 samples in smallest cluster if possible
                    score = float(silhouette_score(X, labels, metric="cosine"))
                    if score > best_score:
                        best_score = score
                        best_k = k

                # If silhouette score is very weak (< 0.11) and min_speakers == 1, fallback to 1 speaker
                if best_score < 0.11 and min_k == 1:
                    best_k = 1

    if best_k == 1:
        raw_labels = np.zeros(len(X), dtype=int)
    else:
        clustering = AgglomerativeClustering(
            n_clusters=best_k,
            metric="cosine",
            linkage="average",
        )
        raw_labels = clustering.fit_predict(X)

    # Order speaker numbers chronologically by appearance (Speaker 1, Speaker 2, ...)
    speaker_map: Dict[int, str] = {}
    next_spk_num = 1
    for lbl in raw_labels:
        lbl_int = int(lbl)
        if lbl_int not in speaker_map:
            speaker_map[lbl_int] = f"Speaker {next_spk_num}"
            next_spk_num += 1

    for idx_in_valid, phrase_idx in enumerate(valid_indices):
        phrases[phrase_idx]["speaker"] = speaker_map[int(raw_labels[idx_in_valid])]

    # Fill any tiny un-embedded phrases with nearest neighbor's speaker
    last_spk = "Speaker 1"
    for p in phrases:
        if "speaker" in p:
            last_spk = p["speaker"]
        else:
            p["speaker"] = last_spk

    # Smooth isolated very short (< 0.65s) single-phrase speaker flips surrounded by the same speaker
    for i in range(1, len(phrases) - 1):
        prev_s = phrases[i - 1]["speaker"]
        curr_s = phrases[i]["speaker"]
        next_s = phrases[i + 1]["speaker"]
        dur = phrases[i]["end"] - phrases[i]["start"]
        if prev_s == next_s and curr_s != prev_s and dur < 0.65:
            phrases[i]["speaker"] = prev_s

    for p in phrases:
        for w in p.get("words", []):
            w["speaker"] = p["speaker"]

    return merge_consecutive_speaker_segments(phrases)


def merge_consecutive_speaker_segments(phrases: List[Dict[str, Any]], max_merge_gap: float = 1.25) -> List[Dict[str, Any]]:
    """Merge consecutive phrases from the same speaker if the pause between them is small."""
    if not phrases:
        return []

    merged = [dict(phrases[0])]
    for p in phrases[1:]:
        prev = merged[-1]
        gap = p["start"] - prev["end"]
        same_speaker = p.get("speaker") == prev.get("speaker")
        combined_dur = p["end"] - prev["start"]

        if same_speaker and gap <= max_merge_gap and combined_dur <= 25.0:
            prev["end"] = p["end"]
            prev["text"] = (prev["text"] + " " + p["text"]).strip()
            prev["words"] = prev.get("words", []) + p.get("words", [])
        else:
            merged.append(dict(p))

    return merged


# ---------------------------------------------------------------------------
# 3. Full WhisperX Transcription + Alignment + Diarization Worker
# ---------------------------------------------------------------------------
def run_transcription_job(
    job_id: str,
    media_path: Path,
    model_name: str,
    language: Optional[str],
    num_speakers: Optional[int],
    min_speakers: int,
    max_speakers: int,
    hf_token: Optional[str],
):
    job = JOBS[job_id]
    wav_path = OUTPUTS_DIR / f"{job_id}_16k.wav"

    try:
        with GPU_LOCK:
            import whisperx

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            batch_size = 8 if device == "cuda" else 4

            # Step 1: Extract 16kHz audio
            job["status"] = "processing"
            job["stage"] = "Extracting audio track from video (FFmpeg)..."
            job["progress"] = 10
            t0 = time.time()
            extract_wav_16k(media_path, wav_path)

            audio = whisperx.load_audio(str(wav_path))
            duration_sec = round(len(audio) / 16000.0, 2)
            job["duration_sec"] = duration_sec

            # Step 2: WhisperX Batched Transcription
            job["stage"] = f"Transcribing speech with WhisperX ({model_name} on {device.upper()})..."
            job["progress"] = 25

            try:
                model = whisperx.load_model(
                    model_name,
                    device,
                    compute_type=compute_type,
                    download_root=str(MODELS_DIR),
                )
            except Exception as e:
                if compute_type == "float16":
                    compute_type = "int8"
                    model = whisperx.load_model(
                        model_name,
                        device,
                        compute_type=compute_type,
                        download_root=str(MODELS_DIR),
                    )
                else:
                    raise e

            transcribe_kwargs = {"batch_size": batch_size}
            if language and language != "auto":
                transcribe_kwargs["language"] = language

            result = model.transcribe(audio, **transcribe_kwargs)
            detected_lang = result.get("language", language or "en")
            job["language"] = detected_lang

            # Free Whisper model from VRAM before loading alignment model
            del model
            free_vram()

            # Step 3: Phoneme-Level Millisecond Word Alignment
            job["stage"] = f"Aligning exact word timestamps ({detected_lang.upper()})..."
            job["progress"] = 60
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_lang,
                    device=device,
                    model_dir=str(MODELS_DIR),
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    device,
                    return_char_alignments=False,
                )
                del model_a
                free_vram()
            except Exception as align_err:
                print(f"[Warning] Word alignment fallback: {align_err}")

            # Step 4: Speaker Diarization (Recognizing Different Voices)
            job["stage"] = "Recognizing & separating speakers (Voice Diarization)..."
            job["progress"] = 80

            used_diarizer = "ECAPA-TDNN Neural Voiceprints (Local)"
            segments_out = []

            if hf_token and hf_token.strip():
                try:
                    from whisperx.diarize import DiarizationPipeline
                    diarize_model = DiarizationPipeline(
                        use_auth_token=hf_token.strip(),
                        device=device,
                    )
                    diarize_kwargs = {}
                    if num_speakers and num_speakers > 0:
                        diarize_kwargs["num_speakers"] = num_speakers
                    else:
                        diarize_kwargs["min_speakers"] = min_speakers
                        diarize_kwargs["max_speakers"] = max_speakers

                    diarize_segments = diarize_model(audio, **diarize_kwargs)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                    del diarize_model
                    free_vram()

                    # Normalize speaker names to Speaker 1, Speaker 2, ...
                    spk_map = {}
                    counter = 1
                    raw_segs = result.get("segments", [])
                    for s in raw_segs:
                        raw_spk = s.get("speaker", "SPEAKER_00")
                        if raw_spk not in spk_map:
                            spk_map[raw_spk] = f"Speaker {counter}"
                            counter += 1
                        s["speaker"] = spk_map[raw_spk]
                        segments_out.append({
                            "start": round(float(s.get("start", 0.0)), 2),
                            "end": round(float(s.get("end", 0.0)), 2),
                            "speaker": s["speaker"],
                            "text": s.get("text", "").strip(),
                            "words": s.get("words", []),
                        })
                    used_diarizer = "Pyannote 3.1 + WhisperX Word Alignment"
                except Exception as pyannote_err:
                    print(f"[Warning] Pyannote failed ({pyannote_err}), falling back to ECAPA-TDNN.")
                    segments_out = []

            if not segments_out:
                diarized = diarize_with_ecapa(
                    wav_path=wav_path,
                    segments=result.get("segments", []),
                    device=device,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                for s in diarized:
                    segments_out.append({
                        "start": round(float(s.get("start", 0.0)), 2),
                        "end": round(float(s.get("end", 0.0)), 2),
                        "speaker": s.get("speaker", "Speaker 1"),
                        "text": s.get("text", "").strip(),
                        "words": s.get("words", []),
                    })

            free_vram()

            elapsed = round(time.time() - t0, 1)
            speakers = sorted(list({s["speaker"] for s in segments_out}))

            job["status"] = "completed"
            job["stage"] = "Completed!"
            job["progress"] = 100
            job["elapsed_sec"] = elapsed
            job["diarizer"] = used_diarizer
            job["speakers"] = speakers
            job["segments"] = segments_out

            # Persist result JSON to outputs folder
            result_json_path = OUTPUTS_DIR / f"{job_id}.json"
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(job, f, ensure_ascii=False, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        job["status"] = "error"
        job["stage"] = f"Error: {str(e)}"
        job["error"] = str(e)
    finally:
        if wav_path.exists():
            try:
                wav_path.unlink()
            except Exception:
                pass
        free_vram()


# ---------------------------------------------------------------------------
# 4. API Routes
# ---------------------------------------------------------------------------
@app.get("/api/system")
async def api_system():
    return get_system_info()


@app.post("/api/transcribe")
async def api_transcribe(
    file: Optional[UploadFile] = File(None),
    local_path: Optional[str] = Form(None),
    model_name: str = Form("large-v3"),
    language: str = Form("auto"),
    num_speakers: Optional[int] = Form(None),
    min_speakers: int = Form(1),
    max_speakers: int = Form(8),
    hf_token: Optional[str] = Form(None),
):
    job_id = str(uuid.uuid4())[:8]

    if local_path and local_path.strip():
        cleaned = local_path.strip().strip('"').strip("'")
        src_path = Path(cleaned)
        if not src_path.exists() or not src_path.is_file():
            raise HTTPException(status_code=400, detail=f"File not found: {cleaned}")
        media_path = src_path
        filename = src_path.name
        media_url = f"/api/media/{job_id}"
    elif file is not None:
        filename = file.filename or f"video_{job_id}.mp4"
        safe_ext = Path(filename).suffix or ".mp4"
        media_path = UPLOADS_DIR / f"{job_id}{safe_ext}"
        with open(media_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        media_url = f"/api/media/{job_id}"
    else:
        raise HTTPException(status_code=400, detail="Please upload a video/audio file or provide a local file path.")

    JOBS[job_id] = {
        "job_id": job_id,
        "filename": filename,
        "media_path": str(media_path),
        "media_url": media_url,
        "model_name": model_name,
        "status": "queued",
        "stage": "Queued...",
        "progress": 2,
        "created_at": time.time(),
        "segments": [],
        "speakers": [],
    }

    worker = threading.Thread(
        target=run_transcription_job,
        args=(
            job_id,
            media_path,
            model_name,
            language,
            num_speakers if (num_speakers and num_speakers > 0) else None,
            min_speakers,
            max_speakers,
            hf_token,
        ),
        daemon=True,
    )
    worker.start()

    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
async def api_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    job = JOBS[job_id]
    return {
        k: v
        for k, v in job.items()
        if k != "media_path"
    }


@app.get("/api/media/{job_id}")
async def api_job_media(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    media_path = Path(JOBS[job_id]["media_path"])
    if not media_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(media_path)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 62)
    print("  WhisperX Studio is starting on http://127.0.0.1:8765")
    print("=" * 62 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")

