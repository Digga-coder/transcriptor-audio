FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero para cachear dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Precargar modelos de WhisperX durante el build
# Esto evita que los usuarios esperen a que se descarguen en runtime
RUN python -c "
import whisperx
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
compute_type = 'float16' if device == 'cuda' else 'int8'

print('Precargando modelos...')
print('1/3 Whisper large-v3...')
whisperx.load_model('large-v3', device=device, compute_type=compute_type)

print('2/3 Whisper medium...')
whisperx.load_model('medium', device=device, compute_type=compute_type)

print('3/3 Modelos precargados correctamente')
" || echo "Advertencia: No se pudieron precargar todos los modelos, se descargaran en runtime"

# Copiar codigo de la aplicacion
COPY . .

# Puerto por defecto de Gradio
EXPOSE 7860

# Variable de entorno para HuggingFace Spaces
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

CMD ["python", "app.py"]