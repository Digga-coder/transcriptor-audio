import os
import re
import sys
import tempfile
import traceback
import subprocess
import csv
from datetime import timedelta
from pathlib import Path
from collections import defaultdict

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import whisperx
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("HUGGINGFACE_TOKEN", ""))

MODEL_CACHE = {}
DEVICE = None
COMPUTE_TYPE = None

# ============ UTILIDADES ============

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
    out_path = os.path.join(tempfile.gettempdir(), f"whisperx_{os.getpid()}_{os.path.basename(input_path)}.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-i", input_path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", out_path],
            capture_output=True, text=True, check=True
        )
        if os.path.exists(out_path):
            return out_path
    except:
        pass
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
    except:
        pass
    return None


# ============ PROCESAMIENTO ============

def transcribe_single(audio_path, model_name, language, num_speakers, hf_token, progress=None):
    """Transcribe un solo archivo y devuelve los segmentos con speakers."""
    wav_path = convert_to_wav(audio_path)
    if not wav_path:
        return None, "Error: No se pudo convertir el audio a WAV."

    device, compute_type = get_device()
    model = get_model(model_name)

    audio = whisperx.load_audio(wav_path)
    result = model.transcribe(audio, language=language if language else None)

    if not result["segments"]:
        return None, "No se detecto voz en el audio."

    # Alineación
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device)

    # Diarización
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(
        wav_path,
        min_speakers=num_speakers if num_speakers > 0 else None,
        max_speakers=num_speakers if num_speakers > 0 else None,
    )
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result, None


def build_transcript_text(result):
    """Genera transcripción formateada con speakers y timestamps."""
    lines = []
    current_speaker = None
    current_text = ""
    current_start = None

    for segment in result["segments"]:
        speaker = segment.get("speaker", "Desconocido")
        text = segment["text"].strip()
        start = segment["start"]

        if speaker != current_speaker:
            if current_speaker is not None:
                lines.append(f"[{format_timestamp(current_start)}] {current_speaker}: {current_text.strip()}")
            current_speaker = speaker
            current_text = text
            current_start = start
        else:
            current_text += " " + text

    if current_speaker is not None:
        lines.append(f"[{format_timestamp(current_start)}] {current_speaker}: {current_text.strip()}")

    return "\n".join(lines)


# ============ GENERACIÓN DE INFORMES ============

def extract_actions_and_decisions(text):
    """Extrae acciones, tareas y decisiones de la transcripción."""
    # Patrones de acción en español e inglés
    action_patterns = [
        r"(?i)(accion[:\s]+|tarea[:\s]+|pendiente[:\s]+|hay que\s+.+|tenemos que\s+.+|vamos a\s+.+|queda en\s+.+|responsable[:\s]+.+|deadline[:\s]+.+|fecha limite[:\s]+.+|para el\s+\d+|proximo\s+\w+|se debe\s+.+|falta\s+.+|urgente[:\s]+.+)",
        r"(?i)(action[:\s]+|task[:\s]+|pending[:\s]+|we need to\s+.+|we have to\s+.+|let.s\s+.+|responsible[:\s]+.+|due[:\s]+.+|by\s+(?:monday|tuesday|wednesday|thursday|friday|next week|tomorrow))",
    ]

    decision_patterns = [
        r"(?i)(acordamos\s+.+|decidimos\s+.+|queda establecido\s+.+|se aprobo\s+.+|se rechazo\s+.+|se acordo\s+.+|conclusion[:\s]+.+|resolucion[:\s]+.+)",
        r"(?i)(we agreed\s+.+|we decided\s+.+|it was approved\s+.+|it was rejected\s+.+|conclusion[:\s]+.+|resolution[:\s]+.+)",
    ]

    actions = []
    decisions = []
    lines = text.split("\n")

    for line in lines:
        line_clean = re.sub(r"\[.*?\]\s*(SPEAKER_\w+|Desconocido):\s*", "", line).strip()
        if not line_clean:
            continue
        for pattern in action_patterns:
            if re.search(pattern, line_clean):
                actions.append(line_clean)
                break
        for pattern in decision_patterns:
            if re.search(pattern, line_clean):
                decisions.append(line_clean)
                break

    return actions, decisions


def generate_executive_summary(text):
    """Genera un resumen ejecutivo con puntos clave."""
    # Limpiar texto
    clean_lines = []
    for line in text.split("\n"):
        clean = re.sub(r"\[.*?\]\s*(SPEAKER_\w+|Desconocido):\s*", "", line).strip()
        if clean and len(clean) > 20:
            clean_lines.append(clean)

    full_text = " ".join(clean_lines)

    # Extraer palabras clave frecuentes (temas)
    words = re.findall(r"\b[A-Za-zÁÉÍÓÚáéíóúÑñ]{4,}\b", full_text.lower())
    stopwords = {"esta", "esto", "para", "como", "pero", "todo", "muy", "bien", "donde", "cuando", "entonces", "porque", "sobre", "entre", "desde", "hasta", "tambien", "después", "antes", "ayer", "hoy", "mañana", "quiero", "puedo", "vamos", "tenemos", "hay", "ser", "estar", "tener", "hacer", "decir", "ir", "ver", "dar", "saber", "pensar", "creer", "parecer", "quedar", "pasar", "llevar", "dejar", "seguir", "encontrar", "llamar", "venir", "poder", "querer", "like", "just", "time", "know", "think", "want", "need", "going", "really", "something", "would", "could", "should"}
    filtered = [w for w in words if w not in stopwords and len(w) > 4]
    freq = defaultdict(int)
    for w in filtered:
        freq[w] += 1
    top_themes = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]

    # Extraer oraciones clave (las más largas suelen ser explicativas)
    sentences = re.split(r"[.!?]", full_text)
    key_sentences = [s.strip() for s in sentences if len(s.strip()) > 60 and len(s.strip()) < 300][:5]

    # Puntos clave basados en transiciones
    key_points = []
    transition_words = ["primero", "segundo", "tercero", "en primer lugar", "además", "por otro lado", "en cuanto a", "respecto a", "finalmente", "en conclusión", "resumiendo", "lo mas importante", "el objetivo", "la meta", "first", "second", "third", "additionally", "furthermore", "regarding", "finally", "in conclusion", "the goal", "objective"]
    for sentence in sentences:
        s_lower = sentence.lower()
        for tw in transition_words:
            if tw in s_lower:
                clean_s = sentence.strip()
                if len(clean_s) > 30:
                    key_points.append(clean_s)
                break

    summary = "=== RESUMEN EJECUTIVO ===\n\n"
    summary += "1. TEMAS PRINCIPALES TRATADOS:\n"
    for theme, count in top_themes:
        summary += f"   - {theme.capitalize()} (mencionado {count} veces)\n"

    summary += "\n2. PUNTOS CLAVE:\n"
    for i, point in enumerate(key_points[:8], 1):
        summary += f"   {i}. {point}\n"

    summary += "\n3. SENTENCIAS RELEVANTES:\n"
    for i, sent in enumerate(key_sentences, 1):
        summary += f"   {i}. {sent}\n"

    return summary


def generate_actions_report(text):
    """Genera informe de acciones y decisiones."""
    actions, decisions = extract_actions_and_decisions(text)

    report = "=== INFORME DE ACCIONES Y DECISIONES ===\n\n"

    report += "📋 ACCIONES PENDIENTES / TAREAS:\n"
    report += "=" * 50 + "\n"
    if actions:
        for i, action in enumerate(actions, 1):
            report += f"{i}. {action}\n"
    else:
        report += "No se detectaron acciones explicitas. Revisar transcripcion completa.\n"

    report += "\n\n✅ DECISIONES TOMADAS / ACUERDOS:\n"
    report += "=" * 50 + "\n"
    if decisions:
        for i, decision in enumerate(decisions, 1):
            report += f"{i}. {decision}\n"
    else:
        report += "No se detectaron decisiones explicitas. Revisar transcripcion completa.\n"

    return report


# ============ EXPORTACIÓN ============

def save_reports(base_name, transcript, summary, actions_report, num_speakers, model_name, language):
    """Guarda todos los informes en archivos temporales."""
    outputs = []

    # 1. Transcripción completa
    t_path = os.path.join(tempfile.gettempdir(), f"{base_name}_TRANSCRIPCION.txt")
    with open(t_path, "w", encoding="utf-8") as f:
        f.write(f"=== TRANSCRIPCION COMPLETA ===\n")
        f.write(f"Archivo: {base_name}\n")
        f.write(f"Modelo: {model_name} | Idioma: {language or 'auto'}\n")
        f.write(f"Interlocutores: {num_speakers}\n\n")
        f.write(transcript)
    outputs.append(t_path)

    # 2. Resumen ejecutivo
    s_path = os.path.join(tempfile.gettempdir(), f"{base_name}_RESUMEN.txt")
    with open(s_path, "w", encoding="utf-8") as f:
        f.write(summary)
    outputs.append(s_path)

    # 3. Acciones
    a_path = os.path.join(tempfile.gettempdir(), f"{base_name}_ACCIONES.txt")
    with open(a_path, "w", encoding="utf-8") as f:
        f.write(actions_report)
    outputs.append(a_path)

    # 4. CSV con datos estructurados
    c_path = os.path.join(tempfile.gettempdir(), f"{base_name}_DATOS.csv")
    with open(c_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tipo", "Contenido"])
        for line in transcript.split("\n"):
            if line.strip():
                speaker = re.search(r"(SPEAKER_\w+|Desconocido)", line)
                speaker_name = speaker.group(1) if speaker else "N/A"
                text = re.sub(r"\[.*?\]\s*(SPEAKER_\w+|Desconocido):\s*", "", line).strip()
                if text:
                    writer.writerow([speaker_name, text])
    outputs.append(c_path)

    return outputs


# ============ INTERFAZ GRADIO ============

def process_batch(files, model_name, language, num_speakers, hf_token, progress=gr.Progress()):
    if not files:
        return "Error: No se ha subido ningun archivo.", [], None

    if not hf_token and HF_TOKEN:
        hf_token = HF_TOKEN

    if not hf_token:
        return (
            "Error: Necesitas un token de HuggingFace para la diarizacion.\n\n"
            "1. Ve a https://huggingface.co/settings/tokens\n"
            "2. Crea un token con permiso 'read'\n"
            "3. Acepta los terminos en pyannote/speaker-diarization-3.1",
            [], None
        )

    all_outputs = []
    full_log = ""

    file_list = files if isinstance(files, list) else [files]
    total = len(file_list)

    for idx, file_obj in enumerate(file_list):
        if hasattr(file_obj, 'name'):
            audio_path = file_obj.name
        elif isinstance(file_obj, str):
            audio_path = file_obj
        else:
            continue

        base_name = Path(audio_path).stem
        progress((idx + 0.5) / total, desc=f"Procesando {base_name}...")

        result, error = transcribe_single(audio_path, model_name, language, num_speakers, hf_token)
        if error:
            full_log += f"❌ {base_name}: {error}\n\n"
            continue

        # Generar informes
        transcript = build_transcript_text(result)
        summary = generate_executive_summary(transcript)
        actions_report = generate_actions_report(transcript)
        detected_speakers = len({s.get("speaker", "Desconocido") for s in result["segments"]})

        # Guardar archivos
        outputs = save_reports(base_name, transcript, summary, actions_report, detected_speakers, model_name, result.get("language", "auto"))
        all_outputs.extend(outputs)

        full_log += f"✅ {base_name} completado ({detected_speakers} interlocutores)\n"

    progress(1.0, desc="Completado!")

    # Crear ZIP con todo
    import zipfile
    zip_path = os.path.join(tempfile.gettempdir(), "informes_reunion.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in all_outputs:
            zf.write(f, os.path.basename(f))

    return full_log, all_outputs, zip_path


MODEL_OPTIONS = ["large-v3", "large-v2", "medium", "small", "base", "tiny"]
LANGUAGE_OPTIONS = ["", "es", "en", "fr", "de", "it", "pt", "ja", "zh", "ko", "ar", "ru"]

with gr.Blocks(title="Transcriptor de Reuniones - Constructora Digital", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        "# Transcriptor Inteligente de Reuniones\n"
        "## Constructora Digital - Analisis completo con IA\n\n"
        "Sube uno o varios audios para obtener:\n"
        "- Transcripcion completa con identificacion de interlocutores\n"
        "- Resumen ejecutivo con temas y puntos clave\n"
        "- Informe de acciones, tareas y decisiones detectadas\n"
        "- Exportacion en TXT y CSV\n"
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Archivos de Audio (1 o varios)",
                file_types=["audio"],
                file_count="multiple",
            )
            model_dropdown = gr.Dropdown(
                choices=MODEL_OPTIONS,
                value="large-v3",
                label="Modelo Whisper",
                info="Large-v3 = maxima calidad con GPU",
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
            hf_token_input = gr.Textbox(
                value=HF_TOKEN,
                label="Token de HuggingFace",
                type="password",
                info="Necesario para diarizacion",
            )
            transcribe_btn = gr.Button("Generar Informes Completos", variant="primary")

        with gr.Column(scale=2):
            log_output = gr.Textbox(
                label="Estado del Procesamiento",
                lines=5,
                max_lines=10,
            )
            files_output = gr.File(
                label="Archivos Generados (descargar individualmente)",
                file_count="multiple",
            )
            zip_output = gr.File(
                label="Descargar Todo (ZIP)",
            )

    transcribe_btn.click(
        fn=process_batch,
        inputs=[file_input, model_dropdown, language_dropdown, num_speakers, hf_token_input],
        outputs=[log_output, files_output, zip_output],
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
