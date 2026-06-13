# -*- coding: utf-8 -*-
"""
Backend del demo (TODA la inferencia corre aquí, en Python).
El front (index.html) solo dibuja y hace fetch a /analizar.

Mejoras aplicadas:
  - Estado por cliente (IP) en vez de global compartido.
  - Motor de respuesta importado desde motor_respuesta.py (sin duplicación).
  - CORS habilitado para red local (otros dispositivos en la misma LAN).
  - Manejo de excepciones en inferencia: si falla el modelo, devuelve 500 con detalle.
  - Health-check endpoint (/health) para diagnóstico rápido.

Uso:
    python app.py
    abrir http://127.0.0.1:5000
"""
import os, sys, time, traceback
from flask import Flask, request, jsonify, send_from_directory

# ---------- CORS (opcional) ----------
try:
    from flask_cors import CORS
    _CORS_OK = True
except ImportError:
    _CORS_OK = False
    print("[WARN] flask-cors no está instalado. CORS deshabilitado.")
    print("       Si necesitas acceso desde otros dispositivos, instala: pip install flask-cors")

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------- Carga de dependencias con manejo de errores ----------
try:
    import numpy as np
except ImportError as e:
    print("[ERROR] numpy no está instalado:", e); sys.exit(1)

try:
    from inferencia import estimar_va
except Exception as e:
    print("[ERROR] No se pudo cargar inferencia.py:", e)
    traceback.print_exc()
    sys.exit(1)

try:
    from motor_respuesta import EstadoCanal, nivel_accion, render, ETA_POR_SEV
except ImportError as e:
    print("[ERROR] No se pudo importar motor_respuesta.py:", e)
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__, static_folder=HERE)
if _CORS_OK:
    CORS(app)  # permite acceso desde otros dispositivos en la red local

# ---------- Estado por cliente (IP) ----------
# En producción usarías session/cookie/JWT. Para demo, IP es suficiente.
CLIENTES = {}  # ip -> EstadoCanal
VENTANA_MIN = 30
MAX_IMPOTENCIA = 2

def _estado_cliente():
    ip = request.remote_addr or "unknown"
    if ip not in CLIENTES:
        CLIENTES[ip] = EstadoCanal(ventana_min=VENTANA_MIN, max_impotencia=MAX_IMPOTENCIA)
    return CLIENTES[ip]

# ---------- Helpers ----------
import random as _rnd

def _ticket(): return f"SDFTC-{_rnd.randint(63000, 63999)}"

def _eta(sev): return f"{10 + int(sev * 35)} min"

# Plantillas empáticas para el demo (fallback cuando motor_respuesta no devuelve nada)
_PLANT = {
    ("frustracion", "validar"): "Vemos tu mensaje y registramos la inconsistencia. Ya estamos revisando la tabla afectada.",
    ("frustracion", "asistir"): "Entendemos la molestia 🙏. La falla ya está identificada (ticket {tk}) y el equipo de plataforma está trabajando. ETA estimada de recarga: {eta}.",
    ("frustracion", "intervenir"): "Lamentamos mucho el impacto 🙏. Confirmamos el incidente (ticket {tk}) y lo priorizamos AHORA. ETA {eta}. Te avisamos en cuanto quede recargada — no necesitas seguir reportando.",
    ("preocupacion", "validar"): "Te leemos. Estamos atentos al cierre; por ahora sigue dentro del SLA.",
    ("preocupacion", "asistir"): "Tranquil@, entendemos la preocupación por tus entregas. La tabla está en recarga (ticket {tk}), ETA {eta}, a tiempo para tu reporte. Te confirmamos apenas finalice.",
    ("preocupacion", "intervenir"): "Sabemos que tienes entregas en juego. Quedó priorizada (ticket {tk}), ETA {eta}. Si el plazo se ajusta, te coordinamos una vía alterna para que no pierdas el día.",
    ("impotencia", "validar"): "Estamos contigo en esto. No tienes que resolverlo sol@: ya lo tomamos nosotros.",
    ("impotencia", "asistir"): "Entendemos que esto no depende de ti. El equipo ya está sobre el problema (ticket {tk}), ETA {eta}. Quedate tranquilo, te mantenemos al tanto sin que tengas que preguntar.",
    ("impotencia", "intervenir"): "Lamentamos la espera prolongada. Escalamos tu caso a un responsable humano para darte seguimiento directo (ticket {tk}). Te contactarán en breve.",
    ("neutral", "ninguno"): "Anotado 👍. Reviso el estado y te aviso si detecto alguna anomalía.",
    ("positivo", "ninguno"): "¡Excelente! 🎉 Me alegra que todo esté funcionando. Cualquier cosa, aquí estamos.",
}

# ---------- Endpoints ----------

@app.route("/health")
def health():
    """Endpoint para verificar que el backend está vivo y el modelo cargado."""
    return jsonify({
        "status": "ok",
        "modelo_cargado": True,
        "motor_respuesta": True,
        "clientes_activos": len(CLIENTES),
    })

@app.route("/analizar", methods=["POST"])
def analizar():
    data = request.get_json(silent=True) or {}
    msg = data.get("mensaje", "").strip()
    if not msg:
        return jsonify({"error": "mensaje vacío"}), 400

    # 1. Inferencia (con manejo de excepciones)
    try:
        r = estimar_va(msg)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "fallo en inferencia", "detalle": str(e)}), 500

    emo = r["emocion"]
    sev = r["severidad"]
    nv = nivel_accion(sev, emo)

    # 2. Estado por cliente + contexto del hilo
    estado = _estado_cliente()
    ts = time.time()
    import datetime as dt
    ts_dt = dt.datetime.fromtimestamp(ts)
    
    # Ajustar severidad según contexto del hilo (mensajes previos del usuario)
    sev_ajustada = estado.ajustar_severidad("demo_user", ts_dt, emo, sev)
    nv = nivel_accion(sev_ajustada, emo)
    
    estado.registrar("demo_user", ts_dt, emo, sev)

    escalar = False
    if emo == "impotencia":
        if nv == "intervenir" or estado.impotencia_sostenida("demo_user", ts_dt):
            nv = "intervenir"
            escalar = True

    # 3. Generar respuesta usando el motor completo
    ctx = {"tabla": "la tabla afectada", "ticket": _ticket(), "eta": _eta(sev_ajustada), "usuario": "equipo", "severidad": sev_ajustada}
    respuesta = render(emo, nv, ctx)
    if not respuesta:
        # fallback a plantillas demo
        t = _PLANT.get((emo, nv))
        if t:
            respuesta = t.replace("{tk}", _ticket()).replace("{eta}", _eta(sev_ajustada))
    if not respuesta:
        respuesta = "Anotado 👍. Reviso el estado y te aviso si detecto alguna anomalía."

    return jsonify({
        "emocion": emo,
        "valencia": r["valencia"],
        "arousal": r["arousal"],
        "severidad_bruta": sev,
        "severidad_ajustada": round(sev_ajustada, 3),
        "nivel": nv,
        "escalado": escalar,
        "respuesta": respuesta,
        "meta": r["meta"],
        "rama_texto": r.get("rama_texto", "desconocida"),
    })

@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(HERE, filename)

if __name__ == "__main__":
    print("=" * 60)
    print("  Demo: Detector de Fricción Emocional")
    print("  Servidor: http://127.0.0.1:5000")
    print("  Health check: http://127.0.0.1:5000/health")
    print("  Presiona Ctrl+C para detener")
    print("=" * 60)
    # host='0.0.0.0' permite acceso desde otros dispositivos en la red
    app.run(host="0.0.0.0", port=5000, debug=False)
