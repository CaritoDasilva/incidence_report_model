import csv, os, joblib, re, numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import torch
from transformers import AutoModel, AutoTokenizer

def cargar_datos(path='dataset_friccion_emocional.csv'):
    """Carga mensaje, valencia, arousal, y grupo (usuario)."""
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

def extraer_embeddings_beto(textos, model, tokenizer, batch_size=16, device='cpu'):
    """Extrae embeddings del [CLS] token para una lista de textos."""
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(textos), batch_size):
            batch = textos[i:i+batch_size].tolist()
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    return np.vstack(embeddings)

def entrenar_beto_ridge(X, y_v, y_a, groups, n_splits=5, alpha=1.0, device='cpu'):
    """Entrena Ridge sobre embeddings BETO con GroupKFold."""
    print("Cargando BETO desde directorio local...")
    tokenizer = AutoTokenizer.from_pretrained('beto')
    model = AutoModel.from_pretrained('beto')
    for param in model.parameters():
        param.requires_grad = False
    print(f"Modelo congelado en {device}")

    print("Extrayendo embeddings de todo el dataset...")
    X_emb = extraer_embeddings_beto(X, model, tokenizer, device=device)

    gkf = GroupKFold(n_splits=n_splits)
    res_v, res_a = [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_emb, y_v, groups), 1):
        X_tr, X_te = X_emb[train_idx], X_emb[test_idx]
        yv_tr, yv_te = y_v[train_idx], y_v[test_idx]
        ya_tr, ya_te = y_a[train_idx], y_a[test_idx]

        ridge_v = Ridge(alpha=alpha)
        ridge_a = Ridge(alpha=alpha)
        ridge_v.fit(X_tr, yv_tr)
        ridge_a.fit(X_tr, ya_tr)

        pv = ridge_v.predict(X_te)
        pa = ridge_a.predict(X_te)

        res_v.append({
            'fold': fold, 'r2': r2_score(yv_te, pv), 'rmse': np.sqrt(mean_squared_error(yv_te, pv))
        })
        res_a.append({
            'fold': fold, 'r2': r2_score(ya_te, pa), 'rmse': np.sqrt(mean_squared_error(ya_te, pa))
        })

    print("\n=== VALIDACIÓN HONESTA BETO (GroupKFold por remitente) ===")
    print(f"Valencia  R²: {np.mean([r['r2'] for r in res_v]):.3f} ± {np.std([r['r2'] for r in res_v]):.3f}")
    print(f"Valencia  RMSE: {np.mean([r['rmse'] for r in res_v]):.3f} ± {np.std([r['rmse'] for r in res_v]):.3f}")
    print(f"Arousal   R²: {np.mean([r['r2'] for r in res_a]):.3f} ± {np.std([r['r2'] for r in res_a]):.3f}")
    print(f"Arousal   RMSE: {np.mean([r['rmse'] for r in res_a]):.3f} ± {np.std([r['rmse'] for r in res_a]):.3f}")

    # Entrenamiento final sobre todo el dataset
    print("\nEntrenando modelo final sobre todo el dataset...")
    ridge_v_final = Ridge(alpha=alpha)
    ridge_a_final = Ridge(alpha=alpha)
    ridge_v_final.fit(X_emb, y_v)
    ridge_a_final.fit(X_emb, y_a)

    # Guardar
    os.makedirs('modelo_beto', exist_ok=True)
    joblib.dump({'ridge_v': ridge_v_final, 'ridge_a': ridge_a_final}, 'modelo_beto/ridges.joblib')
    model.save_pretrained('modelo_beto/beto_model')
    tokenizer.save_pretrained('modelo_beto/beto_tokenizer')
    print("Modelo guardado en modelo_beto/")

    return ridge_v_final, ridge_a_final, model, tokenizer, res_v, res_a

if __name__ == '__main__':
    X, y_v, y_a, groups = cargar_datos()
    print(f"Dataset: {len(X)} mensajes, {len(np.unique(groups))} usuarios únicos")
    entrenar_beto_ridge(X, y_v, y_a, groups)
