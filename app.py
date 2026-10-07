import io
import os

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydub import AudioSegment
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizer

# Lokal o'zbekcha Whisper modeli (katta encoder + yengil decoder) — fp16'da ~1.6GB VRAM,
# 4GB'lik videokartaga ham bemalol sig'adi. Boshqa yo'lga o'tish uchun STT_MODEL_PATH'ni o'zgartiring.
MODEL_PATH = os.environ.get(
    "STT_MODEL_PATH", os.path.join(os.path.dirname(__file__), "model")
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_PATH)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_PATH, language="uz", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH, dtype=dtype).to(device)
model.eval()

prompt_ids = tokenizer.convert_tokens_to_ids(
    ["<|startoftranscript|>", "<|uz|>", "<|transcribe|>", "<|notimestamps|>"]
)

app = FastAPI(title="Uzbek STT")


def decode_audio(raw: bytes) -> np.ndarray:
    audio = AudioSegment.from_file(io.BytesIO(raw))
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= 1 << (8 * audio.sample_width - 1)
    return samples


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "templates", "index.html"))


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    raw = await audio.read()
    samples = decode_audio(raw)

    inputs = feature_extractor(samples, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device).to(dtype)
    decoder_input_ids = torch.tensor([prompt_ids], device=device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=225,
        )

    text = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return JSONResponse({"text": text.strip()})
