# WORK: cómo trabajar con el proyecto

Esta guía describe el procedimiento completo para pasar de los datos originales a un modelo entrenado y exportado.

## 1. Preparar el entorno

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
```

Las dependencias principales son:

- `torch`: tensores, modelo, optimización y DataLoader.
- `torchaudio`: carga de audio, remuestreo, mel-spectrogramas y efectos.
- `numpy`: semillas y comparación de salidas ONNX.
- `wandb`: seguimiento visual de los experimentos.
- `onnx` y `onnxruntime`: exportación y verificación.

El proyecto no necesita `torchcodec` para leer los WAV: usa un lector PCM interno basado en la librería estándar `wave` y `numpy`, porque las versiones recientes de `torchaudio` pueden convertir `torchaudio.load()` en una llamada obligatoria a TorchCodec.

Comprueba que el intérprete seleccionado por VS Code sea el mismo que utilizas en el terminal:

```bash
python --version
python -c "import torch, torchaudio, wandb; print(torch.__version__)"
```

## 2. Extraer y preparar los datos

Descarga el dataset oficial y conserva solo las diez clases:

```bash
python data/download_and_prepare.py \
  --root ./data \
  --output-dir ./data/speech_commands_10
```

Si aparece `SSL: CERTIFICATE_VERIFY_FAILED` por un proxy o antivirus de la red, primero corrige el certificado/CA local. Como alternativa temporal, solo en una red de confianza, puedes usar:

```bash
python data/download_and_prepare.py \
  --root ./data \
  --output-dir ./data/speech_commands_10 \
  --insecure-download
```

La opción desactiva la verificación TLS únicamente durante esta descarga y está desactivada por defecto.

El script realiza estas operaciones:

1. descarga `speech_commands_v0.02.tar.gz` si no existe;
2. extrae el archivo;
3. copia `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop` y `go`;
4. copia las listas oficiales `validation_list.txt` y `testing_list.txt` filtradas;
5. copia `_background_noise_` para noise injection.

La estructura esperada es:

```text
data/speech_commands_10/
├── yes/
├── no/
├── up/
├── down/
├── left/
├── right/
├── on/
├── off/
├── stop/
├── go/
├── _background_noise_/
├── validation_list.txt
└── testing_list.txt
```

Nota: `data/` es la carpeta de descarga original y puede contener todas las clases de Speech Commands. No la uses directamente para entrenar. La carpeta filtrada correcta es `data/speech_commands_10/`; allí solo aparecen las diez clases del proyecto, además de `_background_noise_`.

Si vuelves a ejecutar el comando, las carpetas de clases y ruido se reemplazan para dejar una preparación limpia.

## 3. Construir el caché

Construye los mel-spectrogramas normalizados:

```bash
python train.py \
  --build-cache \
  --data-dir ./data/speech_commands_10 \
  --cache-dir ./data/spectrogram_cache \
  --n-mels 64 \
  --wandb-mode offline
```

El caché hace lo siguiente:

1. carga cada `.wav`;
2. remuestrea a 16 kHz;
3. recorta o rellena a 1 segundo;
4. calcula `MelSpectrogram` con `n_fft=400`, `win_length=400`, `hop_length=160`;
5. convierte amplitud a dB;
6. calcula media y desviación globales;
7. guarda cada espectrograma normalizado como `.pt`;
8. escribe `metadata.json`.

Importante: reconstruye el caché si cambias `--n-mels`, la frecuencia de muestreo o cualquier parámetro de preprocessing. Para `--augment-mode full`, la metadata debe contener `wav_path`; los cachés antiguos producirán un error indicando que deben reconstruirse.

## 4. Cómo se hace el split

El entrenamiento usa `split_dataset(dataset, seed)`.

```text
100% de clips originales
├── 70% train
├── 15% validation
└── 15% test
```

La semilla por defecto es `42`:

```bash
--seed 42
```

El split se hace antes de cualquier aumento. Por ello:

- train puede usar waveform augmentation y SpecAugment;
- validation siempre usa espectrogramas limpios del caché;
- test siempre usa espectrogramas limpios del caché;
- nunca se generan copias aumentadas antes de separar los clips.

Usa la misma semilla para comparar dos entrenamientos:

```bash
python train.py --augment-mode none --seed 42 ...
python train.py --augment-mode full --seed 42 ...
```

## 5. Entrenar un modelo base

Entrenamiento sin aumento:

APY KEY: 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

```bash
python train.py \
  --data-dir ./data/speech_commands_10 \
  --cache-dir ./data/spectrogram_cache \
  --augment-mode none \
  --epochs 20 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --weight-decay 0 \
  --checkpoint-path ./checkpoints/base/best_model.pt \
  --wandb-project speech-commands-10 \
  --wandb-mode online
```

Durante cada época se imprimen:

- `train_loss`;
- `train_acc`;
- `val_loss`;
- `val_acc`;
- `val_f1`.

El checkpoint se reemplaza únicamente cuando mejora `val/f1`.

## 6. Aumentar los datos

### Solo SpecAugment

```bash
python train.py \
  --augment-mode specaugment \
  --freq-mask-param 8 \
  --time-mask-param 16 \
  --num-masks 2 \
  --checkpoint-path ./checkpoints/specaugment/best_model.pt \
  --wandb-mode online
```

SpecAugment opera sobre el mel-spectrograma normalizado. En cada llamada puede enmascarar regiones de frecuencia y tiempo usando la media del espectrograma como valor de reemplazo.

### Audio + SpecAugment

```bash
python train.py \
  --augment-mode full \
  --noise-dir ./data/speech_commands_10/_background_noise_ \
  --snr-min 5 \
  --snr-max 20 \
  --max-shift-ms 100 \
  --p-augment 1.0 \
  --freq-mask-param 8 \
  --time-mask-param 16 \
  --num-masks 2 \
  --checkpoint-path ./checkpoints/augmented/best_model.pt \
  --wandb-mode online
```

El orden es:

```text
waveform original
  -> time shift
  -> noise injection
  -> pitch/time effect opcional
  -> mel-spectrograma
  -> normalización del caché
  -> SpecAugment
  -> modelo
```

Para que parte de los ejemplos permanezca limpia:

```bash
--p-augment 0.8
```

Esto significa que aproximadamente 80% de las lecturas de train se aumentan y 20% se devuelven desde el caché sin aumento.

Los efectos opcionales son:

```bash
--use-pitch-shift
--use-time-stretch
```

Están desactivados por defecto. El segundo usa speed perturbation con remuestreo para mantener el coste bajo; ambos pueden hacer el entrenamiento más lento.

## 7. Ver el entrenamiento en vivo

### W&B online

Inicia sesión una vez:

```bash
wandb login
```

Ejecuta el entrenamiento:

```bash
python train.py --wandb-mode online --run-name model-b-base
```

El terminal muestra el progreso por época. En el panel de W&B se pueden observar en vivo:

- `train/loss` y `val/loss`;
- `train/accuracy` y `val/accuracy`;
- `val/f1`;
- `gap/accuracy`;
- `gap/loss`;
- `learning_rate`;
- parámetros y MACs del modelo.

La brecha de accuracy se calcula como:

```text
train accuracy - validation accuracy
```

Una brecha que crece mucho suele indicar overfitting. Loss alta tanto en train como en validation suele indicar underfitting o un entrenamiento insuficiente.

### W&B offline

Si no tienes conexión:

```bash
python train.py --wandb-mode offline
```

Después sincroniza los runs:

```bash
wandb sync wandb/offline-run-*
```

Para comparación visual, usa el mismo proyecto y nombres de run distintos.

## 8. Scheduler y early stopping

Cosine annealing:

```bash
python train.py \
  --scheduler cosine \
  --epochs 30
```

ReduceLROnPlateau según validation F1:

```bash
python train.py \
  --scheduler plateau \
  --epochs 30
```

Early stopping después de cinco épocas sin mejorar validation F1:

```bash
python train.py \
  --early-stopping-patience 5
```

Se pueden combinar las tres opciones con `--weight-decay`.

## 9. Ver la arquitectura

```bash
python scripts/draw_architecture.py
```

La salida incluye las capas hoja, sus shapes y sus parámetros. También imprime:

```text
total_trainable_parameters=...
estimated_macs=...
```

La red espera entradas con shape:

```text
(batch, 1, 64, 101)
```

## 10. Ejecutar los seis experimentos

El runner usa las tres configuraciones de `configs/experiments.py` tanto para base como para augmented:

Puedes dividir el trabajo en dos ejecuciones. Primero ejecuta los tres modelos base:

```bash
python run_experiments.py \
  --mode base \
  --data-dir ./data/speech_commands_10 \
  --cache-dir ./data/spectrogram_cache \
  --epochs 20 \
  --seed 42 \
  --wandb-mode online
```

Cuando termine, ejecuta los tres modelos aumentados:

```bash
python run_experiments.py \
  --mode augmented \
  --data-dir ./data/speech_commands_10 \
  --cache-dir ./data/spectrogram_cache \
  --noise-dir ./data/speech_commands_10/_background_noise_ \
  --epochs 20 \
  --seed 42 \
  --wandb-mode online
```

La opción `--mode all` conserva el comportamiento anterior y ejecuta los seis seguidos. `results/summary.csv` se actualiza acumulativamente, por lo que la segunda ejecución conserva las tres filas base y añade las tres augmented.

```bash
python run_experiments.py \
  --data-dir ./data/speech_commands_10 \
  --cache-dir ./data/spectrogram_cache \
  --epochs 20 \
  --seed 42 \
  --wandb-mode online
```

Para una prueba corta local:

```bash
python run_experiments.py \
  --epochs 1 \
  --wandb-mode offline
```

Los runs se ejecutan en este orden:

```text
model-b-base-balanced
model-b-base-regularized
model-b-base-compact
model-b-augmented-balanced
model-b-augmented-regularized
model-b-augmented-compact
```

Cada checkpoint queda en:

```text
checkpoints/<run_name>/best_model.pt
```

El resumen final queda en:

```text
results/summary.csv
```

Incluye modo, configuración, mejor validation F1, test accuracy, test F1, gap y número de parámetros. También se publica como `wandb.Table` en un run de resumen.

## 11. Evaluación final

La evaluación final siempre recarga el mejor checkpoint según `val/f1`. Se registran:

- `test/loss`;
- `test/accuracy`;
- `test/f1`;
- matriz de confusión;
- F1 de cada clase.

La matriz se puede consultar en la sección Media/Charts del run de W&B.

## 12. Exportar a ONNX

Después de entrenar:

```bash
python export_onnx.py \
  --checkpoint-path ./checkpoints/base/best_model.pt \
  --cache-dir ./data/spectrogram_cache \
  --onnx-path ./checkpoints/base/model.onnx \
  --seed 42 \
  --num-samples 5
```

El script:

1. carga `model_state_dict` y `config`;
2. reconstruye `MobileNetStyleCNN`;
3. toma ejemplos del test reproducible;
4. exporta con opset 17;
5. ejecuta el modelo con ONNX Runtime;
6. compara los logits de PyTorch y ONNX.

La salida esperada contiene mensajes como:

```text
Sample 1: PyTorch and ONNX outputs match.
```

La tolerancia configurada es `rtol=1e-3` y `atol=1e-4`.

## 13. Flujo completo recomendado

```bash
# 1. Dependencias
python -m pip install -r requirements.txt

# 2. Datos
python data/download_and_prepare.py \
  --root ./data \
  --output-dir ./data/speech_commands_10

# 3. Caché
python train.py \
  --build-cache \
  --augment-mode none \
  --wandb-mode offline

# 4. Prueba visual de arquitectura
python scripts/draw_architecture.py

# 5. Base
python train.py \
  --augment-mode none \
  --checkpoint-path ./checkpoints/base/best_model.pt \
  --wandb-mode online

# 6. Aumentado
python train.py \
  --augment-mode full \
  --noise-dir ./data/speech_commands_10/_background_noise_ \
  --checkpoint-path ./checkpoints/augmented/best_model.pt \
  --wandb-mode online

# 7. ONNX
python export_onnx.py \
  --checkpoint-path ./checkpoints/base/best_model.pt \
  --cache-dir ./data/spectrogram_cache \
  --onnx-path ./checkpoints/base/model.onnx
```

Para el informe, compara primero `best_val_f1`, `test_accuracy`, `test_f1`, `gap` y `parameters` en `results/summary.csv`, y después revisa las curvas y matrices de confusión en W&B.
