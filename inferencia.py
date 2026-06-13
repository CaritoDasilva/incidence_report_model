# -*- coding: utf-8 -*-
"""
Inferencia del detector de friccion emocional (corre en Python).
Version 3: soporta tres ramas de texto:
  - Rama BETO (nuevo): embeddings contextualizados pre-entrenados en español.
    Congelado como feature extractor + Ridge para fine-tuning.
    R² Valencia 0.943, R² Arousal 0.907 (validación honesta por remitente).
  - Rama TF-IDF (original): memoriza n-gramas del dataset. R² con fuga 0.92.
  - Rama Word2Vec (alternativa): embeddings densos, pero corpus insuficiente.

Carga el modelo entrenado y lo combina con priors lexicos para robustez
ante texto libre (out-of-distribution).
"""
import os, math, re, warnings
warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- Diccionario lexico con word boundaries ----------
LEX = {
    "frus": [
        r"\botra vez\b", r"\bde nuevo\b", r"\bnuevamente\b", r"\bno puede ser\b",
        r"\bdesastre\b", r"\bpésimo\b", r"\bpesimo\b", r"\bhorrible\b",
        r"\bharto\b", r"\bcolmo\b", r"\bno funciona\b", r"\bsigue mal\b",
        r"\bsigue sin\b", r"\bno carga\b", r"\bno cargo\b", r"\bno cargó\b",
        r"\bcaída\b", r"\bcaida\b", r"\bcaído\b", r"\bcaídos\b", r"\bcaidos\b",
        r"\bfalla\b", r"\bfalló\b", r"\bfallo\b", r"\berror\b", r"\bmalo\b", r"\bmala\b",
        r"\binconsistente\b", r"\bincompleto\b", r"\bbasta\b", r"\ben serio\b",
        r"\babsurdo\b", r"\bridículo\b", r"\bridiculo\b", r"\bno sirve\b",
        r"\bcansado\b", r"\bcansada\b", r"\bmolesto\b", r"\bmolesta\b",
        r"\brabia\b", r"\benojado\b", r"\benojada\b", r"\bfuriosa\b",
        r"\bfurioso\b", r"\bindignada\b", r"\bindignado\b",
        r"\bsiempre lo mismo\b", r"\bsin funcionar\b", r"\bno me sirve\b",
        r"\bretrasando\b", r"\bestás retrasando\b", r"\bme estás retrasando\b",
        r"\bpero no\b", r"\bpero eso\b",
        r"\bno puedo trabajar\b", r"\bno puedo avanzar\b", r"\bno puedo seguir\b",
        r"\bestoy bloqueado\b", r"\bestoy bloqueada\b", r"\bparado\b", r"\bparada\b",
        r"\btodo caído\b", r"\btodo caido\b", r"\btodo roto\b",
        r"\bqué pasa\b", r"\bque pasa\b", r"\bqué pasa con\b", r"\bque pasa con\b",
    ],
    "preo": [
        r"\bpreocupa\b", r"\bpreocupada\b", r"\bpreocupado\b", r"\bsi no\b",
        r"\bvoy a perder\b", r"\bno alcanzo\b", r"\bno voy a\b", r"\bno va a llegar\b",
        r"\bse retrasa\b", r"\bse retrasan\b", r"\bretraso\b", r"\bretrasos\b",
        r"\ba tiempo\b", r"\bantes del\b", r"\bantes de\b", r"\bentrega\b",
        r"\breporte\b", r"\breportes\b", r"\bcierre\b", r"\bplazo\b",
        r"\bansiedad\b", r"\bintranquilo\b", r"\bintranquila\b",
        r"\bque voy a\b", r"\by si\b", r"\bmediodía\b", r"\bmediodia\b",
        r"\bno me da el tiempo\b", r"\bme da miedo\b", r"\bme preocupa\b",
        r"\bcuanto tiempo falta\b", r"\bcuándo puedo\b", r"\bcuando puedo\b",
        r"\bpara que pueda\b", r"\bpara poder\b",
    ],
    "impo": [
        r"\bno puedo hacer nada\b", r"\bno está en mis manos\b", r"\bno esta en mis manos\b",
        r"\bno depende de mí\b", r"\bno depende de mi\b", r"\bsolo me queda\b",
        r"\bsolo queda\b", r"\bme resigné\b", r"\bme resigne\b", r"\bresignar\b",
        r"\bresignada\b", r"\bya ni\b", r"\bda lo mismo\b", r"\bpara qué\b",
        r"\bpara que\b", r"\bno tengo acceso\b", r"\baguantar\b",
        r"\bno queda otra\b", r"\bcruzar los dedos\b", r"\bimpotente\b",
        r"\bnada que hacer\b", r"\bsolo esperar\b",
        r"\bno puedo hacer nada al respecto\b", r"\bno tengo control\b",
        r"\bme resigno\b", r"\bno se que hacer\b", r"\bno sé qué hacer\b",
    ],
    "posi": [
        r"\bgracias\b", r"\bexcelente\b", r"\bperfecto\b", r"\bgenial\b",
        r"\bimpecable\b", r"\bbuenísimo\b", r"\bbuenazo\b", r"\btremendo\b",
        r"\bgran trabajo\b", r"\bse agradece\b", r"\bresuelto\b", r"\btodo ok\b",
        r"\btodo bien\b", r"\bya quedó\b", r"\bya quedo\b", r"\bsolucionado\b",
        r"\bmil gracias\b", r"\bfuncionando\b", r"\boperativo\b", r"\brestaurado\b",
    ],
}

ANCLA = {"frus": (-0.70, 0.55), "preo": (-0.60, 0.30), "impo": (-0.80, -0.25), "posi": (0.55, 0.15)}

LEX_RE = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in LEX.items()}


def _hits(text, patterns):
    """Cuenta coincidencias de patrones regex con word boundaries en el texto."""
    return sum(1 for p in patterns if p.search(text))


# ---------- Modelos (lazy loading) ----------
_modelo = None

def _cargar_modelo():
    """Carga los modelos entrenados (BETO, TF-IDF, Word2Vec) una sola vez."""
    global _modelo
    if _modelo is not None:
        return _modelo
    
    import joblib, numpy as np
    from gensim.models import Word2Vec
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    M = {"beto": None, "tfidf": None, "w2v": None, "ridge_w2v": None, "feats": []}
    
    # 1. Cargar BETO (prioridad alta)
    beto_dir = os.path.join(_HERE, "modelo_beto")
    if os.path.exists(beto_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(beto_dir, "beto_tokenizer"))
            model = AutoModel.from_pretrained(os.path.join(beto_dir, "beto_model"))
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            ridges = joblib.load(os.path.join(beto_dir, "ridges.joblib"))
            M["beto"] = {
                "tokenizer": tokenizer,
                "model": model,
                "ridge_v": ridges["ridge_v"],
                "ridge_a": ridges["ridge_a"],
            }
            print("[INFO] BETO cargado correctamente")
        except Exception as e:
            print(f"[WARN] No se pudo cargar BETO: {e}")
    
    # 2. Cargar TF-IDF baseline
    path_tfidf = os.path.join(_HERE, "modelo_friccion_baseline.joblib")
    if os.path.exists(path_tfidf):
        try:
            data = joblib.load(path_tfidf)
            M["tfidf"] = {
                "tfidf": data["tfidf"],
                "valencia_txt": data["valencia_txt"],
                "arousal_txt": data["arousal_txt"],
                "arousal_beh": data["arousal_beh"],
                "valencia_beh": data["valencia_beh"],
                "decision": data["decision"],
                "feats": data["feats"],
            }
            M["feats"] = data["feats"]
        except Exception as e:
            print(f"[WARN] No se pudo cargar TF-IDF: {e}")
    
    # 3. Cargar Word2Vec + Ridge
    path_w2v = os.path.join(_HERE, "modelo_word2valencia.model")
    path_ridge_w2v = os.path.join(_HERE, "ridge_word2vec.joblib")
    if os.path.exists(path_w2v) and os.path.exists(path_ridge_w2v):
        try:
            M["w2v"] = Word2Vec.load(path_w2v)
            M["ridge_w2v"] = joblib.load(path_ridge_w2v)
        except Exception as e:
            print(f"[WARN] No se pudo cargar Word2Vec: {e}")
    
    _modelo = M
    return M


def _embed_beto(text, tokenizer, model, device='cpu'):
    """Extrae embedding del [CLS] de BETO para un texto."""
    import torch
    import numpy as np
    model.to(device)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()


def _embed_w2v(text, w2v, dim=100):
    """Embedding promedio de Word2Vec para un texto."""
    import numpy as np
    words = [w.strip() for w in text.lower().split() if w.strip()]
    vecs = [w2v.wv[w] for w in words if w in w2v.wv]
    if not vecs:
        return np.zeros(dim)
    return np.mean(vecs, axis=0)


# ---------- Meta-features ----------
def meta_features(text, gap_seg=30, espera_seg=0):
    n = len(text) or 1
    excl = text.count("!"); interr = text.count("?")
    letters = [c for c in text if c.isalpha()]
    caps = sum(1 for c in letters if c.isupper())
    cr = caps / len(letters) if letters else 0.0
    return {
        "longitud_msg": n,
        "densidad_exclamacion": round(excl / n * 100, 3),
        "densidad_interrogacion": round(interr / n * 100, 3),
        "ratio_mayusculas": round(cr, 3),
        "seg_desde_msg_previo": gap_seg,
        "es_repregunta": int("?" in text and gap_seg < 120),
        "seg_espera_sin_respuesta_equipo": espera_seg,
    }


# ---------- Inferencia ----------
def estimar_va(text, gap_seg=30, espera_seg=0, rama="beto"):
    """
    Estima (valencia, arousal, emocion, severidad) para un mensaje.
    
    Args:
        rama: "beto" (default), "tfidf", "word2vec", o "auto" (prueba BETO, fallback TF-IDF).
    """
    import numpy as np
    t = text.lower()
    mf = meta_features(text, gap_seg, espera_seg)
    
    # 1. Prior lexico (siempre disponible)
    h = {k: _hits(t, LEX_RE[k]) for k in LEX_RE}
    dom = max(h, key=h.get) if max(h.values()) > 0 else None
    if dom:
        v_lex, a_lex = ANCLA[dom]
    else:
        v_lex, a_lex = 0.0, 0.0
    
    # 2. Cargar modelos
    M = _cargar_modelo()
    
    # 3. Rama de texto (valencia + arousal)
    v_txt, a_txt = 0.0, 0.0
    rama_texto = "ninguna"
    
    if rama in ("beto", "auto") and M["beto"] is not None:
        try:
            emb = _embed_beto(text, M["beto"]["tokenizer"], M["beto"]["model"])
            v_txt = float(M["beto"]["ridge_v"].predict(emb.reshape(1, -1))[0])
            a_txt = float(M["beto"]["ridge_a"].predict(emb.reshape(1, -1))[0])
            rama_texto = "beto"
        except Exception as e:
            if rama == "beto":
                print(f"[WARN] BETO falló: {e}")
            rama_texto = "ninguna"
    
    if rama_texto == "ninguna" and rama in ("tfidf", "auto") and M["tfidf"] is not None:
        X = M["tfidf"]["tfidf"].transform([text])
        v_txt = float(M["tfidf"]["valencia_txt"].predict(X)[0])
        a_txt = float(M["tfidf"]["arousal_txt"].predict(X)[0])
        rama_texto = "tfidf"
    
    if rama_texto == "ninguna" and rama in ("word2vec", "auto") and M["w2v"] is not None:
        emb = _embed_w2v(text, M["w2v"], 100)
        v_txt = float(M["ridge_w2v"]["ridge_v"].predict(emb.reshape(1, -1))[0])
        a_txt = float(M["ridge_w2v"]["ridge_a"].predict(emb.reshape(1, -1))[0])
        rama_texto = "word2vec"
    
    if rama_texto == "ninguna" and M["tfidf"] is None and M["w2v"] is None and M["beto"] is None:
        # Fallback: solo prior lexico
        v = v_lex; a = a_lex
        emo = "neutral"
        sev = min(1.0, math.hypot(v, a) / math.sqrt(2))
        return {
            "emocion": emo, "valencia": round(v, 3), "arousal": round(a, 3),
            "severidad": round(sev, 3), "meta": mf, "hits": h,
            "dom_lexico": dom, "fallback": True, "rama_texto": "lexico",
        }
    
    # 4. Rama conductual (arousal) - siempre disponible si TF-IDF carga
    a_beh = 0.0
    if M["tfidf"] is not None:
        fv = np.array([[mf[f] for f in M["feats"]]])
        a_beh = float(M["tfidf"]["arousal_beh"].predict(fv)[0])
    
    # 5. Fusion
    if dom:
        # Ajustar peso según cantidad de hits léxicos
        total_hits = h[dom]
        if total_hits >= 2:
            w_lex = 0.85  # Múltiples hits: léxico domina fuertemente
        elif total_hits >= 1:
            w_lex = 0.70  # Un hit: peso original
        else:
            w_lex = 0.70
        w_txt = 1.0 - w_lex
        
        v = w_txt * v_txt + w_lex * v_lex
        a = 0.55 * a_beh + 0.45 * a_lex
        
        # Floor de arousal según emoción léxica detectada
        if dom == "frus":
            # Frustración: arousal mínimo según intensidad del mensaje
            min_a = 0.40
            # Bonus por signos de exclamación y mayúsculas
            excl_bonus = min(0.15, text.count("!") * 0.05)
            caps_bonus = min(0.10, mf["ratio_mayusculas"] * 0.3)
            min_a = min(0.75, min_a + excl_bonus + caps_bonus)
            # Bonus por múltiples hits léxicos
            if total_hits >= 2:
                min_a = min(0.75, min_a + 0.10)
            a = max(a, min_a)
        elif dom == "preo":
            a = max(a, 0.20)  # Preocupación: arousal mínimo moderado
        elif dom == "impo":
            a = min(a, 0.15)  # Impotencia: arousal máximo bajo (resignación)
    else:
        v = 0.6 * v_txt
        a = a_beh if M["tfidf"] is not None else a_txt
    
    v = float(np.clip(v, -1, 1))
    a = float(np.clip(a, -1, 1))
    
    # 6. Capa de decision
    emo = "neutral"
    if M["tfidf"] is not None:
        emo = str(M["tfidf"]["decision"].predict([[v, a]])[0])
    elif dom:
        emo = {"frus": "frustracion", "preo": "preocupacion", "impo": "impotencia", "posi": "positivo"}.get(dom, "neutral")
    
    sev = min(1.0, math.hypot(v, a) / math.sqrt(2))
    
    return {
        "emocion": emo, "valencia": round(v, 3), "arousal": round(a, 3),
        "severidad": round(sev, 3), "meta": mf, "hits": h,
        "dom_lexico": dom, "fallback": False, "rama_texto": rama_texto,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("COMPARACION: BETO vs TF-IDF vs Word2Vec")
    print("=" * 60)
    tests = [
        "otra vez la tabla de ventas con datos malos?? esto ya paso ayer!!",
        "ESTOY ENOJADA, el reporte no carga y nadie dice NADA!!!",
        "si no carga antes del mediodia se me retrasan todos los informes",
        "ya no puedo hacer nada, solo me queda esperar a que lo arreglen",
        "excelente, justo lo que necesitaba, mil gracias!",
        "hola, una consulta sobre la carga de hoy",
        "notificacion: la carga normal termino sin problemas",
    ]
    for m in tests:
        r_beto = estimar_va(m, rama="beto")
        r_tfidf = estimar_va(m, rama="tfidf")
        r_w2v = estimar_va(m, rama="word2vec")
        print(f"\n{m[:50]}")
        print(f"  BETO    : {r_beto['emocion']:12s} v={r_beto['valencia']:+.2f} a={r_beto['arousal']:+.2f} sev={r_beto['severidad']:.2f}")
        print(f"  TF-IDF  : {r_tfidf['emocion']:12s} v={r_tfidf['valencia']:+.2f} a={r_tfidf['arousal']:+.2f} sev={r_tfidf['severidad']:.2f}")
        print(f"  Word2Vec: {r_w2v['emocion']:12s} v={r_w2v['valencia']:+.2f} a={r_w2v['arousal']:+.2f} sev={r_w2v['severidad']:.2f}")
