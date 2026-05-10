# Transcriptor de Audio con Diarizacion

Aplicacion web para transcribir archivos de audio identificando automaticamente a cada interlocutor.

## Caracteristicas

- **Transcripcion** con OpenAI Whisper (via WhisperX)
- **Diarizacion** con pyannote.audio (identifica quien habla y cuando)
- **Alineacion** de palabras a nivel de timestamp
- **Interfaz web** con Gradio
- **Soporte multi-idioma** (auto-deteccion o manual)
- **Descarga** de transcripcion en formato texto

## Opciones de despliegue gratuito recomendadas

### Opcion 1: Hugging Face Spaces (Recomendada)

**Gratis, GPU disponible, siempre online.**

1. Crea una cuenta en [huggingface.co](https://huggingface.co)
2. Acepta los terminos de uso:
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
3. Crea un token de acceso en [Settings > Tokens](https://huggingface.co/settings/tokens)
4. Crea un nuevo Space: [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Docker**
   - Hardware: **CPU Basic** (gratis) o **ZeroGPU** (2h/dia GPU gratis)
5. En **Settings > Secrets** del Space, crea:
   - `HF_TOKEN` = tu token de HuggingFace
6. Sube el codigo (ver seccion Git mas abajo)

**Ventajas:**
- 100% gratuito
- Modelos de ML alojados en la misma infraestructura
- URLs propias: `https://tu-usuario-tu-space.hf.space`
- ZeroGPU te da acceso gratuito a GPU A100 (2h/dia)

### Opcion 2: Oracle Cloud Free Tier

**VM siempre gratis, nunca expira.**

1. Crea cuenta en [Oracle Cloud](https://www.oracle.com/cloud/free/)
2. Crea una instancia **VM.Standard.A1.Flex** (4 OCPU, 24GB RAM - siempre gratis)
3. Conectate por SSH, instala Docker y despliega:

```bash
# En la VM
git clone https://github.com/TU-USUARIO/transcriptor-audio.git
cd transcriptor-audio
docker build -t transcriptor .
docker run -d -p 80:7860 -e HF_TOKEN=tu_token transcriptor
```

**Ventajas:**
- Siempre online, nunca duerme
- CPU potente (ARM Ampere)
- Tu propia infraestructura

### Opcion 3: Render

**Gratis pero se duerme tras 15 min de inactividad.**

1. Crea cuenta en [render.com](https://render.com)
2. New Web Service > Build from Git repo
3. Configuracion:
   - **Runtime:** Docker
   - **Plan:** Free
   - **Environment Variable:** `HF_TOKEN`

**Desventaja:** Se duerme tras 15 min, tarda ~30s en despertar.

### Opcion 4: Local (tu PC)

```bash
# Requisitos: Python 3.10+, ffmpeg
pip install -r requirements.txt
# Crear .env con HF_TOKEN=tu_token
python app.py
# Abrir http://localhost:7860
```

## Subir a GitHub

```bash
# En la carpeta del proyecto
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/transcriptor-audio.git
git push -u origin main
```

**IMPORTANTE:** El archivo `.env` esta en `.gitignore` para no subir tu token.

## Arquitectura

```
Usuario
  |
  v
Gradio Web UI (Python/Flask)
  |
  v
WhisperX Pipeline
  |-- Whisper (transcripcion)
  |-- CTC Forced Alignment (timestamps de palabra)
  |-- pyannote.audio (diarizacion - quien habla)
  |
  v
Salida: Transcripcion con timestamps y speakers
```

**Componentes:**
- **Frontend:** Gradio (HTML/CSS/JS auto-generado)
- **Backend:** Python 3.10
- **ML:** PyTorch + WhisperX + pyannote.audio
- **Audio:** ffmpeg + soundfile
- **Container:** Docker (Linux)

**Flujo de datos:**
1. Usuario sube audio (MP3, WAV, M4A, etc.)
2. ffmpeg convierte a WAV 16kHz mono
3. Whisper transcribe a texto
4. Modelo CTC alinea palabras con timestamps exactos
5. pyannote identifica segmentos de cada hablante
6. Se combina transcripcion + speakers
7. Resultado formateado con timestamps y etiquetas SPEAKER_00, SPEAKER_01...

## Tokens y Permisos necesarios

Para usar la diarizacion (identificar interlocutores) necesitas:

1. **HuggingFace Token** con permisos de `read`
2. Haber aceptado los terminos de:
   - `pyannote/speaker-diarization-3.1`
   - `pyannote/segmentation-3.0`

Sin token, la transcripcion funciona pero NO identifica interlocutores.

## Modelos disponibles

| Modelo | Precision | Velocidad | VRAM necesaria |
|--------|-----------|-----------|----------------|
| large-v3 | Alta | Lenta | ~10GB |
| large-v2 | Alta | Lenta | ~10GB |
| medium | Media | Media | ~5GB |
| small | Media | Rapida | ~2GB |
| base | Baja | Muy rapida | ~1GB |
| tiny | Baja | Ultra rapida | ~0.5GB |

## Limitaciones

- **Duracion:** Audio largo (>1h) requiere mas RAM/VRAM
- **Formatos:** Cualquier formato soportado por ffmpeg
- **Idiomas:** 99 idiomas ( Whisper multilingue)
- **Speakers:** Funciona mejor con 2-10 interlocutores claros

## Licencia

MIT

## Autor

Creado para uso personal y profesional.