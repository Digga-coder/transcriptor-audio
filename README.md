# Transcriptor de Audio con Diarizacion

Aplicacion web para transcribir archivos de audio identificando automaticamente a cada interlocutor.

## Caracteristicas

- **Transcripcion** con OpenAI Whisper (via WhisperX)
- **Diarizacion** con pyannote.audio (identifica quien habla y cuando)
- **Alineacion** de palabras a nivel de timestamp
- **Interfaz web** con Gradio
- **Soporte multi-idioma** (auto-deteccion o manual)
- **Descarga** de transcripcion en formato texto

## Despliegue en Hugging Face Spaces (Recomendado)

### 1. Crea el Space

1. Ve a [huggingface.co/new-space](https://huggingface.co/new-space)
2. **Space name:** `transcriptor-audio`
3. **SDK:** `Gradio`
4. **Hardware:** `CPU Basic` (gratis, siempre online) o `ZeroGPU` (GPU gratis 2h/dia)
5. Crea el Space

### 2. Configura el token

1. En tu Space, ve a **Settings > Secrets**
2. Añade una variable:
   - **Name:** `HF_TOKEN`
   - **Value:** tu token de HuggingFace
3. Guarda

### 3. Sube el codigo

En la pestaña **Files** de tu Space:
- Sube `app.py`
- Sube `requirements.txt`
- Sube `.gitattributes`

O conecta tu cuenta de GitHub y selecciona este repositorio.

### 4. Reinicia el Space

Ve a **Settings > Factory Reboot** para que se instalen las dependencias.

**URL final:** `https://TU-USUARIO-transcriptor-audio.hf.space`

## Opciones de hardware

| Hardware | Coste | GPU | Tiempo limite | Recomendacion |
|----------|-------|-----|---------------|---------------|
| **CPU Basic** | Gratis | No | Ninguno | **Recomendado** para uso continuo |
| **ZeroGPU** | Gratis | A100 | 60-180s/request | Para audios cortos y modelos pequenos |
| **GPU T4** | Pago | T4 | Ninguno | Para uso profesional |

**Nota sobre ZeroGPU:** Tiene timeout. Para audios largos (>10 min) con `large-v3`, usa **CPU Basic**. Para audios cortos (<5 min) con modelos `small` o `base`, ZeroGPU es mas rapido.

## Uso local

```bash
# Requisitos: Python 3.10+, ffmpeg
pip install -r requirements.txt

# Crea un archivo .env con tu token:
echo "HF_TOKEN=tu_token_aqui" > .env

python app.py
# Abre http://localhost:7860
```

## Tokens y Permisos necesarios

Para usar la diarizacion (identificar interlocutores) necesitas:

1. **HuggingFace Token** con permisos de `read`
2. Haber aceptado los terminos de:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

Sin token, la transcripcion funciona pero NO identifica interlocutores.

## Modelos disponibles

| Modelo | Precision | Velocidad CPU | Velocidad GPU | VRAM |
|--------|-----------|---------------|---------------|------|
| large-v3 | Alta | Lenta | Media | ~10GB |
| large-v2 | Alta | Lenta | Media | ~10GB |
| medium | Media | Media | Rapida | ~5GB |
| small | Media | Rapida | Muy rapida | ~2GB |
| base | Baja | Muy rapida | Ultra rapida | ~1GB |
| tiny | Baja | Ultra rapida | Instantanea | ~0.5GB |

**Recomendacion CPU Basic:** `small` o `base`
**Recomendacion ZeroGPU:** `small` o `medium`

## Arquitectura

```
Usuario
  |
  v
Gradio Web UI (Python)
  |
  v
WhisperX Pipeline
  |-- Whisper (transcripcion)
  |-- CTC Forced Alignment (timestamps palabra)
  |-- pyannote.audio (diarizacion - quien habla)
  |
  v
Salida: Transcripcion con timestamps y speakers
```

**Stack tecnico:**
- **Frontend:** Gradio (auto-generado)
- **Backend:** Python 3.10
- **ML:** PyTorch + WhisperX + pyannote.audio
- **Audio:** ffmpeg + soundfile
- **Hosting:** Hugging Face Spaces

## Licencia

MIT