---
title: Transcriptor de Audio con Diarizacion
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
---

# Transcriptor de Audio con Identificacion de Interlocutores

Aplicacion web para transcribir archivos de audio identificando automaticamente a cada interlocutor.

## Caracteristicas

- **Transcripcion** con OpenAI Whisper (via WhisperX)
- **Diarizacion** con pyannote.audio (identifica quien habla y cuando)
- **Alineacion** de palabras a nivel de timestamp
- **Interfaz web** con Gradio
- **Soporte multi-idioma** (auto-deteccion o manual)
- **Descarga** de transcripcion en formato texto

## Uso

1. Sube un archivo de audio (MP3, WAV, M4A, etc.)
2. Selecciona modelo e idioma
3. Introduce tu token de HuggingFace (necesario para diarizacion)
4. Haz clic en "Transcribir"
5. Descarga la transcripcion como .txt

## Requisitos previos

Debes haber aceptado los terminos de uso en:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

## Modelos disponibles

| Modelo | Precision | Velocidad CPU | VRAM |
|--------|-----------|---------------|------|
| large-v3 | Alta | Lenta | ~10GB |
| large-v2 | Alta | Lenta | ~10GB |
| medium | Media | Media | ~5GB |
| small | Media | Rapida | ~2GB |
| base | Baja | Muy rapida | ~1GB |
| tiny | Baja | Ultra rapida | ~0.5GB |

## Stack tecnico

- **Frontend:** Gradio
- **Backend:** Python 3.10
- **ML:** PyTorch + WhisperX + pyannote.audio
- **Audio:** ffmpeg + soundfile

## Licencia

MIT