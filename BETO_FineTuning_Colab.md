# BETO Fine-Tuning en Google Colab (GPU Gratuita)

Este notebook fine-tunea BETO (`dccuchile/bert-base-spanish-wwm-cased`) para detectar valencia y arousal en mensajes de fricción emocional.

## Instrucciones

1. Ve a [Google Colab](https://colab.research.google.com/)
2. Crea un nuevo notebook (File → New Notebook)
3. Copia y pega las celdas de abajo
4. Ejecuta en orden (Runtime → Run all, o Ctrl+F9)
5. Asegúrate de que GPU esté activada: Edit → Notebook settings → Hardware accelerator → GPU

---

## Celda 1: Verificar GPU

```python
!nvidia-smi
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

---

## Celda 2: Instalar dependencias

```python
!pip install transformers scikit-learn pandas -q
```

---

## Celda 3: Montar Google Drive (para guardar el modelo)

```python
from google.colab import drive
drive.mount('/content/drive')

# Crear carpeta para guardar el modelo
!mkdir -p /content/drive/MyDrive/beto_finetuned
```

---

## Celda 4: Crear el dataset (300 mensajes)

Copia tu archivo `dataset_friccion_emocional.csv` a Google Drive, o crea uno simple aquí:

```python
import csv

# Dataset simplificado (reemplaza con tu dataset completo)
# O bien: sube dataset_friccion_emocional.csv a Drive y cárgalo:
# df = pd.read_csv('/content/drive/MyDrive/dataset_friccion_emocional.csv')

# Ejemplo con datos mínimos para probar:
dataset = [
    ("otra vez la tabla sales_daily con datos inconsistentes", -0.70, 0.55, "USR_01"),
    ("alguien me explica por que sigue sin cargar", -0.75, 0.65, "USR_01"),
    ("si no carga antes del mediodia se me retrasan todos los informes", -0.60, 0.30, "USR_01"),
    ("de mi lado no puedo hacer nada", -0.88, -0.33, "USR_01"),
    ("excelente justo lo que necesitaba", 0.55, 0.15, "USR_02"),
    ("hola vi algo raro en orders", -0.07, 0.04, "USR_03"),
    ("esto no cuadra los numeros estan mal", -0.65, 0.55, "USR_03"),
    ("ya no se que mas hacer solo me queda esperar", -0.80, -0.35, "USR_03"),
    ("gracias por todo", 0.50, 0.20, "USR_04"),
    ("todo esta caido no puedo trabajar", -0.70, 0.50, "USR_05"),
]

# Guardar como CSV
with open('dataset_friccion_emocional.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id','fecha','remitente','rol','mensaje','longitud_msg',
                     'densidad_exclamacion','densidad_interrogacion','ratio_mayusculas',
                     'seg_desde_msg_previo','es_repregunta','seg_espera_sin_respuesta_equipo',
                     'valencia','arousal','emocion_dominante','cuadrante_russell',
                     'severidad','friccion'])
    for i, (msg, v, a, user) in enumerate(dataset, 1):
        sev = (v**2 + a**2)**0.5 / (2**0.5)
        emo = 'frustracion' if v < -0.5 and a > 0.4 else 'preocupacion' if v < -0.5 and a > 0.2 else 'impotencia' if v < -0.5 and a < 0.2 else 'positivo' if v > 0.3 else 'neutral'
        writer.writerow([i, f'2026-05-19T12:{i:02d}:00', user, 'usuario', msg, len(msg),
                          0, 0, 0, 60, 0, 0, v, a, emo, 'NO', sev, 1 if emo != 'neutral' else 0])

print("Dataset creado: dataset_friccion_emocional.csv")
print(f"Mensajes: {len(dataset)}")
```

**Nota**: Para usar tu dataset real de 300 mensajes, súbelo a Google Drive y cárgalo:
```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/dataset_friccion_emocional.csv')
```

---

## Celda 5: Fine-tuning de BETO

```python
import csv, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BETO_MODEL = 'dccuchile/bert-base-spanish-wwm-cased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 15
LR = 2e-5
WARMUP = 0
DROPOUT = 0.1
NUM_CAPAS_CONGELADAS = 8  # Congelar 8 de 12 capas

print(f"Device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LR}")

class FriccionDataset(Dataset):
    def __init__(self, textos, y_valencia, y_arousal, tokenizer, max_len=128):
        self.textos = textos
        self.y_v = y_valencia
        self.y_a = y_arousal
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.textos)

    def __getitem__(self, idx):
        text = self.textos[idx]
        encoding = self.tokenizer(
            text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'y_valencia': torch.tensor(self.y_v[idx], dtype=torch.float),
            'y_arousal': torch.tensor(self.y_a[idx], dtype=torch.float),
        }

class BETORegressor(nn.Module):
    def __init__(self, beto_model, dropout=0.1):
        super().__init__()
        self.beto = beto_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.beto(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits[:, 0], logits[:, 1]

def cargar_datos(path='dataset_friccion_emocional.csv'):
    X, y_v, y_a, groups = [], [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            msg = row['mensaje'].strip()
            if not msg:
                continue
            X.append(msg)
            y_v.append(float(row['valencia']))
            y_a.append(float(row['arousal']))
            groups.append(row['remitente'])
    return np.array(X), np.array(y_v), np.array(y_a), np.array(groups)

def congelar_capas_bajas(model, num_capas_congeladas=8):
    for param in model.embeddings.parameters():
        param.requires_grad = False
    for i, layer in enumerate(model.encoder.layer):
        freeze = i < num_capas_congeladas
        for param in layer.parameters():
            param.requires_grad = not freeze
    print(f"Capas 0-{num_capas_congeladas-1} congeladas, capas {num_capas_congeladas}-11 entrenables")

def entrenar_fold(model, train_loader, val_loader, device, epochs, lr, verbose=True):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP, num_training_steps=total_steps)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y_v = batch['y_valencia'].to(device)
            y_a = batch['y_arousal'].to(device)

            optimizer.zero_grad()
            pred_v, pred_a = model(input_ids, attention_mask)
            loss = criterion(pred_v, y_v) + criterion(pred_a, y_a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_v, all_a, all_pv, all_pa = [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pred_v, pred_a = model(input_ids, attention_mask)
                loss = criterion(pred_v, batch['y_valencia'].to(device)) + criterion(pred_a, batch['y_arousal'].to(device))
                val_loss += loss.item()
                all_v.extend(batch['y_valencia'].numpy())
                all_a.extend(batch['y_arousal'].numpy())
                all_pv.extend(pred_v.cpu().numpy())
                all_pa.extend(pred_a.cpu().numpy())

        r2_v = r2_score(all_v, all_pv)
        r2_a = r2_score(all_a, all_pa)
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        if verbose:
            print(f"  Epoch {epoch+1}/{epochs} | train={avg_train:.4f} | val={avg_val:.4f} | R2v={r2_v:.3f} | R2a={r2_a:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model

# Cargar datos
X, y_v, y_a, groups = cargar_datos()
print(f"\nDataset: {len(X)} mensajes, {len(np.unique(groups))} usuarios\n")

# Cargar BETO
print("Cargando BETO...")
tokenizer = AutoTokenizer.from_pretrained(BETO_MODEL)
beto_base = AutoModel.from_pretrained(BETO_MODEL)
congelar_capas_bajas(beto_base, NUM_CAPAS_CONGELADAS)

# Contar parametros
total_params = sum(p.numel() for p in beto_base.parameters())
trainable_params = sum(p.numel() for p in beto_base.parameters() if p.requires_grad)
print(f"Parametros totales: {total_params:,}")
print(f"Parametros entrenables: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

# Validacion cruzada
print(f"\n{'='*50}")
print("VALIDACION CRUZADA (GroupKFold por remitente)")
print(f"{'='*50}")

gkf = GroupKFold(n_splits=5)
resultados = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_v, groups), 1):
    print(f"\n--- Fold {fold}/5 ---")
    X_train, X_val = X[train_idx], X[val_idx]
    yv_train, yv_val = y_v[train_idx], y_v[val_idx]
    ya_train, ya_val = y_a[train_idx], y_a[val_idx]

    train_dataset = FriccionDataset(X_train, yv_train, ya_train, tokenizer)
    val_dataset = FriccionDataset(X_val, yv_val, ya_val, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    beto_copy = AutoModel.from_pretrained(BETO_MODEL)
    congelar_capas_bajas(beto_copy, NUM_CAPAS_CONGELADAS)
    model = BETORegressor(beto_copy).to(DEVICE)

    start = time.time()
    model = entrenar_fold(model, train_loader, val_loader, DEVICE, epochs=EPOCHS, lr=LR)
    elapsed = time.time() - start
    print(f"  Tiempo: {elapsed:.1f}s")

    model.eval()
    all_v, all_a, all_pv, all_pa = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            pred_v, pred_a = model(input_ids, attention_mask)
            all_v.extend(batch['y_valencia'].numpy())
            all_a.extend(batch['y_arousal'].numpy())
            all_pv.extend(pred_v.cpu().numpy())
            all_pa.extend(pred_a.cpu().numpy())

    r2_v = r2_score(all_v, all_pv)
    r2_a = r2_score(all_a, all_pa)
    rmse_v = np.sqrt(mean_squared_error(all_v, all_pv))
    rmse_a = np.sqrt(mean_squared_error(all_a, all_pa))
    print(f"  R2_v={r2_v:.3f} | R2_a={r2_a:.3f} | RMSE_v={rmse_v:.3f} | RMSE_a={rmse_a:.3f}")
    resultados.append({'fold': fold, 'r2_v': r2_v, 'r2_a': r2_a, 'rmse_v': rmse_v, 'rmse_a': rmse_a})

# Resumen
print(f"\n{'='*50}")
print("RESUMEN")
print(f"{'='*50}")
r2_v_mean = np.mean([r['r2_v'] for r in resultados])
r2_v_std = np.std([r['r2_v'] for r in resultados])
r2_a_mean = np.mean([r['r2_a'] for r in resultados])
r2_a_std = np.std([r['r2_a'] for r in resultados])
print(f"Valencia  R2: {r2_v_mean:.3f} +/- {r2_v_std:.3f}")
print(f"Arousal   R2: {r2_a_mean:.3f} +/- {r2_a_std:.3f}")
print(f"\nCongelado + Ridge:  R2=0.943/0.907")
print(f"Fine-tuned + Head:  R2={r2_v_mean:.3f}/{r2_a_mean:.3f}")
```

---

## Celda 6: Entrenamiento final y guardado

```python
# Entrenar sobre todo el dataset
print("\n--- Entrenamiento final (100% datos) ---")
full_dataset = FriccionDataset(X, y_v, y_a, tokenizer)
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

beto_final = AutoModel.from_pretrained(BETO_MODEL)
congelar_capas_bajas(beto_final, NUM_CAPAS_CONGELADAS)
model_final = BETORegressor(beto_final).to(DEVICE)

model_final = entrenar_fold(model_final, full_loader, full_loader, DEVICE, epochs=20, lr=LR)

# Guardar en Google Drive
save_path = '/content/drive/MyDrive/beto_finetuned'
!mkdir -p {save_path}
torch.save(model_final.state_dict(), f'{save_path}/beto_finetuned.pt')
tokenizer.save_pretrained(f'{save_path}/tokenizer')
print(f"\n✅ Modelo guardado en Google Drive: {save_path}")
print("  - beto_finetuned.pt (weights)")
print("  - tokenizer/ (vocabulario)")

# También guardar en local para descargar
!mkdir -p /content/beto_finetuned
torch.save(model_final.state_dict(), '/content/beto_finetuned/beto_finetuned.pt')
tokenizer.save_pretrained('/content/beto_finetuned/tokenizer')

# Descargar como zip
import shutil
shutil.make_archive('/content/beto_finetuned', 'zip', '/content/beto_finetuned')
from google.colab import files
files.download('/content/beto_finetuned.zip')
print("\n📦 Descargando beto_finetuned.zip...")
```

---

## Celda 7: Prueba del modelo fine-tuned

```python
# Cargar modelo fine-tuned
model_prueba = BETORegressor(AutoModel.from_pretrained(BETO_MODEL)).to(DEVICE)
model_prueba.load_state_dict(torch.load('/content/beto_finetuned/beto_finetuned.pt'))
model_prueba.eval()

pruebas = [
    "otra vez la tabla de ventas con datos malos",
    "estoy harto, todo caido, no puedo trabajar",
    "excelente, justo lo que necesitaba",
    "hola, buenos dias",
    "si no carga antes del mediodia se me retrasan todos los informes",
]

for text in pruebas:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        v, a = model_prueba(inputs['input_ids'], inputs['attention_mask'])
    v = v.cpu().item()
    a = a.cpu().item()
    sev = (v**2 + a**2)**0.5 / (2**0.5)
    emo = 'frustracion' if v < -0.5 and a > 0.4 else 'preocupacion' if v < -0.5 and a > 0.2 else 'impotencia' if v < -0.5 and a < 0.2 else 'positivo' if v > 0.3 else 'neutral'
    print(f"{text[:50]}... → {emo} v={v:+.2f} a={a:+.2f} sev={sev:.2f}")
```

---

## Pasos finales

1. **Ejecuta todas las celdas** (Ctrl+F9 o Runtime → Run all)
2. **Espera 5-15 minutos** (depende de la GPU asignada)
3. **Descarga** el archivo `beto_finetuned.zip` que se genera automáticamente
4. **Descomprime** en tu PC y reemplaza la carpeta `modelo_beto/` en tu workspace

---

## Notas importantes

- **Sesión de Colab**: Si se desconecta, el progreso se pierde. Asegúrate de guardar en Drive.
- **GPU gratuita**: Puede ser T4 (rápido) o K80 (más lento). Si te asignan K80, reinicia la sesión (Runtime → Restart session) hasta que te den T4.
- **Dataset real**: Reemplaza la celda 4 con tu dataset real de 300 mensajes subiéndolo a Drive.
