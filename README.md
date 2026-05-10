---
title: Transcriptor de Audio con Diarizacion
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
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
- **Autenticacion OAuth** con Hugging Face (sin tokens manuales)

## Uso

1. Inicia sesion con tu cuenta de Hugging Face (boton arriba)
2. Sube un archivo de audio (MP3, WAV, M4A, etc.)
3. Selecciona modelo e idioma
4. Haz clic en "Transcribir"
5. Descarga la transcripcion como .txt

## Requisitos previos

Debes haber aceptado los terminos de uso en:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

## Modelos disponibles

| Modelo | Precision | Velocidad CPU | Velocidad GPU | VRAM |
|--------|-----------|---------------|---------------|------|
| large-v3 | Alta | Lenta | Media | ~10GB |
| large-v2 | Alta | Lenta | Media | ~10GB |
| medium | Media | Media | Rapida | ~5GB |
| small | Media | Rapida | Muy rapida | ~2GB |
| base | Baja | Muy rapida | Ultra rapida | ~1GB |
| tiny | Baja | Ultra rapida | Instantanea | ~0.5GB |

## Stack tecnico

- **Frontend:** Gradio
- **Backend:** Python 3.10
- **ML:** PyTorch + WhisperX + pyannote.audio
- **Audio:** ffmpeg + soundfile

## Licencia

MIT