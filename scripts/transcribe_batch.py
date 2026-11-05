#!/usr/bin/env python3
"""Batch transcribe audio directly into authentic Liberian Pidgin English using Gemini Transcribe Live."""
import asyncio
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/mnt/volume_d2wey28/projects/afrispeech-synth/src")

from afrispeech_synth.env import load as load_env
from afrispeech_synth.transcribe import read_audio_pcm
from google import genai
from google.genai import types

load_env("/mnt/volume_d2wey28/projects/afrispeech-synth/.env")

AUDIO_DIR = "/mnt/volume_d2wey28/projects/liberian-english-speech"
OUT_DIR = os.path.join(AUDIO_DIR, "transcripts")
MANIFEST_JSONL = os.path.join(AUDIO_DIR, "transcripts.jsonl")
CONCURRENCY = 2
MAX_RETRIES = 5
MODEL = "models/gemini-3.5-transcribe-live"

LIBERIAN_VOCABULARY = [
    "na", "dey", "dem", "wetin", "plenty", "dis", "dat",
    "quick quick", "giddem", "sabi", "keh", "kreyol", "sef",
    "God don bless", "woman dem", "bebe", "troway", "pikin",
]

SYSTEM_INSTRUCTION = (
    "You are an expert transcriber of Liberian English (Vernacular Liberian English / Liberian Pidgin). "
    "Transcribe the speech verbatim into authentic Liberian English text. "
    "Do not standardize or convert into standard British or American English. "
    "Preserve Liberian vernacular phrasing, particles (such as na, dey, sef, keh, wetin, dem), "
    "and speech patterns exactly as spoken. "
    "Include standard punctuation (commas, periods, question marks)."
)

os.makedirs(OUT_DIR, exist_ok=True)


async def transcribe_audio_live(client: genai.Client, file_path: str, timeout: float = 300.0) -> tuple[str, float]:
    raw_pcm, rate = read_audio_pcm(file_path, default_rate=16000)
    duration = len(raw_pcm) / (rate * 2) if rate else 0
    effective_timeout = max(timeout, duration + 45.0)

    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.TEXT],
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM_INSTRUCTION)]),
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=["en-LR", "lir"],
            mode=types.AudioTranscriptionConfigMode.VERBATIM,
            custom_vocabulary=LIBERIAN_VOCABULARY,
        ),
    )

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        chunk_size = int(rate * 2 * 0.1)
        audio_done = asyncio.Event()

        async def _send():
            for i in range(0, len(raw_pcm), chunk_size):
                chunk = raw_pcm[i : i + chunk_size]
                await session.send_realtime_input(
                    audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={rate}")
                )
                await asyncio.sleep(0.01)
            await session.send_realtime_input(audio_stream_end=True)
            audio_done.set()

        parts = []

        async def _receive():
            async for msg in session.receive():
                sc = msg.server_content
                if not sc:
                    continue
                if sc.input_transcription and sc.input_transcription.text:
                    parts.append(sc.input_transcription.text)
                if audio_done.is_set() and (sc.generation_complete or sc.turn_complete):
                    break

        sender_task = asyncio.create_task(_send())
        receiver_task = asyncio.create_task(_receive())
        try:
            await asyncio.wait_for(asyncio.gather(sender_task, receiver_task), timeout=effective_timeout)
        except asyncio.TimeoutError:
            sender_task.cancel()
            receiver_task.cancel()

        transcript = " ".join(parts).strip()
        return transcript, duration


async def worker(file_path: str, client: genai.Client, sem: asyncio.Semaphore, lock: asyncio.Lock, counters: dict):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    txt_path = os.path.join(OUT_DIR, f"{stem}.txt")

    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 10:
        counters["skipped"] += 1
        return

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0 = time.time()
                transcript, duration = await transcribe_audio_live(client, file_path)
                elapsed = round(time.time() - t0, 2)

                if not transcript:
                    raise RuntimeError("Live ASR returned empty transcript")

                # Save transcript text file
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(transcript + "\n")

                record = {
                    "id": stem,
                    "audio_file": os.path.basename(file_path),
                    "transcript": transcript,
                    "audio_duration_sec": round(duration, 2),
                    "elapsed_sec": elapsed,
                }

                # Append to JSONL manifest
                async with lock:
                    with open(MANIFEST_JSONL, "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counters["done"] += 1
                    done = counters["done"]
                    skipped = counters["skipped"]
                    total = counters["total"]
                    pct = (done + skipped) / total * 100
                    print(f"[{done + skipped}/{total}] ({pct:.1f}%) "
                          f"Transcribed {stem} (audio: {duration:.1f}s, took: {elapsed:.1f}s)", flush=True)
                    print(f"   {transcript[:90]}...", flush=True)
                return

            except Exception as exc:
                if attempt == MAX_RETRIES:
                    async with lock:
                        counters["failed"] += 1
                        print(f"ERROR: {stem} failed after {attempt} retries: {exc}", flush=True)
                    return
                delay = min(30.0, (2 ** attempt) * 1.5)
                print(f"[{stem}] retry {attempt}/{MAX_RETRIES} in {delay:.1f}s ({exc})", flush=True)
                await asyncio.sleep(delay)


async def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment or .env file.")

    client = genai.Client(api_key=api_key)
    files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.mp3")))
    print(f"Found {len(files)} audio files in {AUDIO_DIR}")
    print(f"Model: {MODEL} (Gemini Live API strictly)")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Manifest: {MANIFEST_JSONL}\n", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    counters = {"done": 0, "failed": 0, "skipped": 0, "total": len(files), "start": time.time()}

    tasks = [worker(f, client, sem, lock, counters) for f in files]
    await asyncio.gather(*tasks)

    total_time = time.time() - counters["start"]
    done = counters["done"]
    skipped = counters["skipped"]
    failed = counters["failed"]
    print(f"\nFinished! Done: {done}, Skipped: {skipped}, Failed: {failed} "
          f"in {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
