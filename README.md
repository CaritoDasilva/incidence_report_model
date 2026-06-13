# Detector de Fricción Emocional

Sistema de computación afectiva para detectar frustración, preocupación e impotencia en canales de incidentes de migración de datos.

## Arquitectura

- **BETO congelado + Ridge**: Embeddings contextualizados en español (Valencia R²=0.943, Arousal R²=0.907)
- **Léxico español**: 60+ patrones para cobertura de dominio
- **Contexto del hilo**: Ajuste de severidad según historial del usuario
- **Motor de respuesta**: 4 niveles de intervención (observar → validar → asistir → intervenir)

## Fine-tuning en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CaritoDasilva/incidence_report_model/blob/main/BETO_FineTuning.ipynb)

Haz clic en el badge de arriba para abrir el notebook de fine-tuning directamente en Google Colab con GPU gratuita.

## Estructura del proyecto

| Archivo | Descripción |
|---------|-------------|
| `inferencia.py` | Inferencia con 3 ramas (BETO/TF-IDF/Word2Vec) |
| `motor_respuesta.py` | Motor de respuesta proactiva con contexto de hilo |
| `app.py` | Demo Flask con telemetría en vivo |
| `presentacion.html` | 13 slides de presentación |
| `BETO_FineTuning.ipynb` | Notebook para fine-tuning en Colab |
| `beto_train.py` | Script de entrenamiento BETO congelado |

## Requisitos

```bash
pip install torch transformers scikit-learn flask
```

## Uso

```python
from inferencia import estimar_va

resultado = estimar_va("está todo caído no puedo trabajar")
print(resultado['emocion'])  # frustracion
print(resultado['severidad'])  # 0.60
```

## Autor

Carito Da Silva - Tesis de Magíster en Ingeniería en Informática
