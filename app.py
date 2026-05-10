import os
import sys
import tempfile
import traceback
import subprocess
from datetime import timedelta
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import whisperx

# Token de admin (para el Space owner) - opcional si se configura en Secrets
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN", ""))

MODEL_CACHE = {}
DEVICE = None
COMPUTE_TYPE = None


def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def get_device():
    global DEVICE, COMPUTE_TYPE
    if DEVICE is None:
        if torch.cuda.is_available():
            DEVICE = "cuda"
            COMPUTE_TYPE = "float16"
        else:
            DEVICE = "cpu"
            COMPUTE_TYPE = "int8"
    return DEVICE, COMPUTE_TYPE


def get_model(model_name):
    device, compute_type = get_device()
    key = f"{model_name}_{device}_{compute_type}"
    if key not in MODEL_CACHE:
        print(f"[INFO] Cargando modelo {model_name} en {device}...")
        MODEL_CACHE[key] = whisperx.load_model(
            model_name, device=device, compute_type=compute_type
        )
        print(f"[INFO] Modelo {model_name} cargado.")
    return MODEL_CACHE[key]


def convert_to_wav(input_path):
    if not input_path or not os.path.exists(input_path):
        return None

    input_path = str(Path(input_path).resolve())
    out_path = os.path.join(tempfile.gettempdir(), f"whisperx_{os.getpid()}.wav")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel", "error",
                "-i", input_path,
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                out_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if os.path.exists(out_path):
            return out_path
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg error: {e.stderr}")
    except FileNotFoundError:
        print("[WARN] ffmpeg no encontrado, usando fallback")

    try:
        data, sr = sf.read(input_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != 16000:
            import scipy.signal
            num_samples = int(len(data) * 16000 / sr)
            data = scipy.signal.resample(data, num_samples)
        sf.write(out_path, data.astype(np.float32), 16000)
        if os.path.exists(out_path):
            return out_path
    except Exception as e:
        print(f"[WARN] soundfile fallback error: {e}")

    return None


def transcribe_audio(request: gr.Request, file_obj, model_name, language, num_speakers):
    if file_obj is None:
        return "Error: No se ha subido ningun archivo.", None

    # Extraer ruta del archivo
    if hasattr(file_obj, 'name'):
        audio_path = file_obj.name
    elif isinstance(file_obj, str):
        audio_path = file_obj
    else:
        return f"Error: Tipo de archivo no soportado: {type(file_obj)}", None

    if not os.path.exists(audio_path):
        return f"Error: El archivo no existe: {audio_path}", None

    # Obtener token OAuth del usuario autenticado
    user_token = None
    if request and hasattr(request, 'headers') and request.headers.get("authorization"):
        user_token = request.headers.get("authorization").replace("Bearer ", "")
    
    # Fallback a token de admin (Secrets) si no hay usuario autenticado
    hf_token = user_token if user_token else HF_TOKEN

    if not hf_token:
        return (
            "Error: Necesitas iniciar sesion con Hugging Face para usar la diarizacion.\n\n"
            "Haz clic en el boton 'Iniciar sesion con Hugging Face' arriba y acepta los permisos.\n\n"
            "Tambien debes aceptar los terminos de uso en:\n"
            "- https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "- https://huggingface.co/pyannote/segmentation-3.0",
            None,
        )

    try:
        wav_path = convert_to_wav(audio_path)
        if wav_path is None:
            return "Error: No se pudo convertir el audio a WAV.", None

        device, compute_type = get_device()
        model = get_model(model_name)

        audio = whisperx.load_audio(wav_path)
        result = model.transcribe(audio, language=language if language else None)

        if not result["segments"]:
            return "No se detecto voz en el audio.", None

        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device
        )

        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=device
        )

        diarize_segments = diarize_model(
            wav_path,
            min_speakers=num_speakers if num_speakers > 0 else None,
            max_speakers=num_speakers if num_speakers > 0 else None,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)

        transcript_lines = []
        current_speaker = None
        current_text = ""
        current_start = None

        for segment in result["segments"]:
            speaker = segment.get("speaker", "Desconocido")
            text = segment["text"].strip()
            start = segment["start"]

            if speaker != current_speaker:
                if current_speaker is not None:
                    transcript_lines.append(
                        f"[{format_timestamp(current_start)}] {current_speaker}: {current_text.strip()}"
                    )
                current_speaker = speaker
                current_text = text
                current_start = start
            else:
                current_text += " " + text

        if current_speaker is not None:
            transcript_lines.append(
                f"[{format_timestamp(current_start)}] {current_speaker}: {current_text.strip()}"
            )

        full_transcript = "\n".join(transcript_lines)

        num_detected = len(
            {s.get("speaker", "Desconocido") for s in result["segments"]}
        )
        summary = "=== Resumen ===\n"
        summary += f"Interlocutores detectados: {num_detected}\n"
        summary += f"Modelo: {model_name}\n"
        summary += f"Idioma: {result.get('language', 'auto')}\n"
        summary += f"Dispositivo: {device}\n\n"

        final_output = summary + full_transcript

        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt", mode="w", encoding="utf-8"
        )
        tmp.write(final_output)
        tmp.close()

        return final_output, tmp.path

    except Exception:
        return f"Error durante la transcripcion:\n{traceback.format_exc()}", None


MODEL_OPTIONS = [
    "large-v3",
    "large-v2",
    "medium",
    "small",
    "base",
    "tiny",
]

LANGUAGE_OPTIONS = [
    "",
    "es",
    "en",
    "fr",
    "de",
    "it",
    "pt",
    "ja",
    "zh",
    "ko",
    "ar",
    "ru",
]

with gr.Blocks(title="Transcriptor de Audio con Diarizacion") as app:
    gr.Markdown(
        "# Transcriptor de Audio con Identificacion de Interlocutores\n\n"
        "Sube un archivo de audio (MP3, WAV, M4A, etc.) para obtener la transcripcion completa "
        "identificando cada interlocutor."
    )

    with gr.Row():
        gr.LoginButton("Iniciar sesion con Hugging Face", variant="primary")
        gr.LogoutButton("Cerrar sesion", variant="secondary")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Archivo de Audio",
                file_types=["audio"],
            )
            model_dropdown = gr.Dropdown(
                choices=MODEL_OPTIONS,
                value="large-v3",
                label="Modelo Whisper",
                info="Modelos mas grandes = mejor precision, mas lento",
            )
            language_dropdown = gr.Dropdown(
                choices=LANGUAGE_OPTIONS,
                value="",
                label="Idioma (vacio = auto)",
            )
            num_speakers = gr.Number(
                value=0,
                label="Numero de interlocutores (0 = auto)",
                precision=0,
            )
            transcribe_btn = gr.Button("Transcribir", variant="primary")

        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="Transcripcion",
                lines=25,
                max_lines=50,
            )
            output_file = gr.File(label="Descargar transcripcion (.txt)")

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[
            file_input,
            model_dropdown,
            language_dropdown,
            num_speakers,
        ],
        outputs=[output_text, output_file],
    )

if __name__ == "__main__":
    app.launch()