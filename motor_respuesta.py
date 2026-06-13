# -*- coding: utf-8 -*-
"""
Motor de respuesta proactiva (Avance 1 - Salida del sistema).
Entrada : (emocion, valencia, arousal, severidad, contexto del hilo)
Logica  : umbral de severidad + emocion -> nivel de accion
Salida  : (1) mensaje empatico proactivo  (2) telemetria estructurada
          (3) bandera de escalado a humano (impotencia sostenida)
"""
import pandas as pd, random, math, json, datetime as dt

# ---------- Niveles de accion por severidad ----------
# severidad = magnitud del vector (v,a) normalizada en [0,1]
UMBRALES = {
    "silencio":   0.30,   # < : no se interviene (ruido neutral)
    "validar":    0.50,   # [0.30,0.50): reconocimiento breve
    "asistir":    0.70,   # [0.50,0.70): validacion + estado real + ETA
    # >= 0.70      : intervencion alta (validacion fuerte + ETA + priorizacion)
}

def nivel_accion(severidad, emocion):
    if emocion in ("neutral", "positivo"):
        return "ninguno"
    if severidad < UMBRALES["silencio"]:
        return "observar"
    if severidad < UMBRALES["validar"]:
        return "validar"
    if severidad < UMBRALES["asistir"]:
        return "asistir"
    return "intervenir"

# ---------- Plantillas empaticas por emocion x nivel ----------
# {tab}=tabla, {tk}=ticket, {eta}=ETA estimada, {nm}=nombre/id usuario
PLANTILLAS = {
 ("frustracion","validar"): [
    "Vemos tu mensaje, {nm}. Registramos la inconsistencia en {tab} y ya la estamos revisando.",
 ],
 ("frustracion","asistir"): [
    "Entendemos la molestia, {nm}. La falla en {tab} esta identificada (ticket {tk}); "
    "el equipo de plataforma ya esta trabajando. ETA estimada de recarga: {eta}.",
 ],
 ("frustracion","intervenir"): [
    "Lamentamos el impacto, {nm}. Confirmamos incidente en {tab} (ticket {tk}) y lo priorizamos AHORA. "
    "ETA {eta}. Te avisamos en cuanto quede recargada, no necesitas seguir reportando.",
 ],
 ("preocupacion","validar"): [
    "Te leemos, {nm}. Estamos atentos al cierre de {tab}; aun esta dentro del SLA.",
 ],
 ("preocupacion","asistir"): [
    "Tranquilo {nm}, entendemos la preocupacion por tus entregas. {tab} esta en recarga (ticket {tk}), "
    "ETA {eta}, a tiempo para tu reporte. Te confirmamos apenas finalice.",
 ],
 ("preocupacion","intervenir"): [
    "Sabemos que tienes entregas en juego, {nm}. {tab} (ticket {tk}) quedo priorizada, ETA {eta}. "
    "Si el plazo se ajusta, te coordinamos una via alterna para que no pierdas el dia.",
 ],
 ("impotencia","validar"): [
    "Estamos contigo en esto, {nm}. No tienes que resolverlo solo: ya lo tomamos nosotros.",
 ],
 ("impotencia","asistir"): [
    "{nm}, entendemos que esto no depende de ti. El equipo ya esta sobre {tab} (ticket {tk}), "
    "ETA {eta}. Quedate tranquilo, te mantenemos informado sin que tengas que preguntar.",
 ],
 ("impotencia","intervenir"): [
    "{nm}, lamentamos la espera prolongada. Escalamos tu caso a un responsable humano para darte "
    "seguimiento directo de {tab} (ticket {tk}). Te contactaran en breve.",
 ],
}

ETA_POR_SEV = lambda s: f"{int(10 + s*40)} min"   # mas severidad -> mas atencion (ETA mas corta no; aqui solo ilustrativo)

def render(emocion, nivel, ctx):
    plant = PLANTILLAS.get((emocion, nivel))
    if not plant: return None
    t = plant[0]
    return (t.replace("{tab}", ctx.get("tabla","la tabla afectada"))
             .replace("{tk}", ctx.get("ticket","SDFTC-XXXXX"))
             .replace("{eta}", ctx.get("eta", ETA_POR_SEV(ctx.get("severidad",0.5))))
             .replace("{nm}", ctx.get("usuario","equipo")))

# ---------- Estado por usuario (impotencia sostenida + contexto de severidad) ----------
class EstadoCanal:
    def __init__(self, ventana_min=60, max_impotencia=2):
        self.hist = {}            # usuario -> lista (timestamp, emocion, severidad)
        self.ventana = ventana_min
        self.max_imp = max_impotencia

    def registrar(self, usuario, ts, emocion, severidad):
        self.hist.setdefault(usuario, []).append((ts, emocion, severidad))

    def impotencia_sostenida(self, usuario, ts):
        lim = ts - dt.timedelta(minutes=self.ventana)
        imp = [1 for (t,e,s) in self.hist.get(usuario,[]) if e=="impotencia" and t>=lim]
        return len(imp) >= self.max_imp

    def contexto_negativo(self, usuario, ts, emocion_actual, severidad_actual):
        """
        Devuelve estadísticas del contexto ANTERIOR al mensaje actual:
        - num_neg: cuántos mensajes negativos en la ventana (excluyendo actual)
        - emociones_previas: lista de emociones antes del actual
        - severidad_max: máxima severidad previa (excluyendo actual)
        """
        lim = ts - dt.timedelta(minutes=self.ventana)
        # Excluir el mensaje actual (último en el historial para este usuario)
        historial = self.hist.get(usuario, [])
        reciente = [(t,e,s) for (t,e,s) in historial if t>=lim and not (t==ts and e==emocion_actual and abs(s-severidad_actual)<0.001)]
        negativas = {"frustracion", "preocupacion", "impotencia"}
        num_neg = sum(1 for (_,e,_) in reciente if e in negativas)
        emociones_previas = [e for (_,e,_) in reciente]
        severidad_max = max((s for (_,_,s) in reciente), default=0.0)
        return {"num_neg": num_neg, "emociones_previas": emociones_previas,
                "severidad_max": severidad_max, "total_mensajes": len(reciente)}

    def ajustar_severidad(self, usuario, ts, emocion, severidad):
        """
        Ajusta severidad según contexto del hilo (mensajes anteriores del usuario):
        - +0.10 si ya hay 1+ emocion negativa previa
        - +0.15 si ya hay 2+ emociones negativas previas
        - +0.20 si hay impotencia previa + nueva frustracion (recuperacion)
        - +0.10 si severidad previa era >= 0.60 (escalada emocional)
        - Cap: max 0.95
        """
        ctx = self.contexto_negativo(usuario, ts, emocion, severidad)
        num_neg = ctx["num_neg"]
        emo_prev = ctx["emociones_previas"]
        sev_max = ctx["severidad_max"]

        bonus = 0.0
        if num_neg >= 2:
            bonus += 0.15
        elif num_neg >= 1:
            bonus += 0.10

        # Impotencia previa + nueva frustracion = recuperacion emocional
        if emocion == "frustracion" and "impotencia" in emo_prev:
            bonus += 0.20

        # Severidad previa alta = escalada
        if sev_max >= 0.60:
            bonus += 0.10

        return min(0.95, severidad + bonus)


# ---------- Motor principal ----------
def responder(mensaje_evt, estado: EstadoCanal):
    """
    mensaje_evt: dict con id,fecha,usuario,emocion,valencia,arousal,severidad,tabla,ticket
    Devuelve dict con respuesta, telemetria y escalado.
    """
    emo = mensaje_evt["emocion"]; sev = mensaje_evt["severidad"]
    ts = dt.datetime.fromisoformat(mensaje_evt["fecha"])
    usuario = mensaje_evt["usuario"]
    estado.registrar(usuario, ts, emo, sev)

    # Ajustar severidad por contexto del hilo (antes de decidir nivel)
    sev_ajustada = estado.ajustar_severidad(usuario, ts, emo, sev)

    nivel = nivel_accion(sev_ajustada, emo)
    escalar = False
    # Regla de derivacion humana: impotencia (alta) o impotencia sostenida
    if emo == "impotencia" and (nivel == "intervenir" or estado.impotencia_sostenida(usuario, ts)):
        nivel = "intervenir"; escalar = True

    ctx = {"tabla": mensaje_evt.get("tabla"), "ticket": mensaje_evt.get("ticket"),
           "usuario": usuario, "severidad": sev_ajustada,
           "eta": ETA_POR_SEV(sev_ajustada)}
    respuesta = render(emo, nivel, ctx) if nivel not in ("ninguno","observar") else None

    telemetria = {
        "ref_msg_id": mensaje_evt["id"],
        "timestamp": mensaje_evt["fecha"],
        "usuario_anon": usuario,
        "valencia": round(mensaje_evt["valencia"],3),
        "arousal": round(mensaje_evt["arousal"],3),
        "emocion_dominante": emo,
        "severidad_bruta": round(sev,3),
        "severidad_ajustada": round(sev_ajustada,3),
        "nivel_accion": nivel,
        "respuesta_emitida": bool(respuesta),
        "escalado_humano": escalar,
    }
    return {"respuesta": respuesta, "telemetria": telemetria, "escalado": escalar}

if __name__ == "__main__":
    import pandas as pd, random
    random.seed(7)
    df = pd.read_csv("dataset_friccion_emocional.csv")
    estado = EstadoCanal(ventana_min=120, max_impotencia=2)
    TABLAS=["transaction","orders","sales_daily","payment_method_detail","stock_movements"]
    salidas=[]
    for _,r in df.iterrows():
        evt={"id":int(r.id),"fecha":r.fecha,"usuario":r.remitente,
             "emocion":r.emocion_dominante,"valencia":float(r.valencia),
             "arousal":float(r.arousal),"severidad":float(r.severidad),
             "tabla":random.choice(TABLAS),"ticket":f"SDFTC-{random.randint(63000,63999)}"}
        out=responder(evt, estado)
        salidas.append(out)

    # Exportar telemetria + respuestas
    tele=[o["telemetria"] for o in salidas]
    pd.DataFrame(tele).to_csv("telemetria_emocional.csv",index=False,encoding="utf-8-sig")
    resp=[{"ref_msg_id":o["telemetria"]["ref_msg_id"],
           "emocion":o["telemetria"]["emocion_dominante"],
           "severidad":o["telemetria"]["severidad"],
           "nivel":o["telemetria"]["nivel_accion"],
           "escalado":o["escalado"],
           "respuesta_bot":o["respuesta"]} for o in salidas if o["respuesta"]]
    with open("respuestas_generadas.json","w",encoding="utf-8") as f:
        json.dump(resp,f,ensure_ascii=False,indent=2)

    # Resumen
    from collections import Counter
    niveles=Counter(t["nivel_accion"] for t in tele)
    print("Total eventos:", len(tele))
    print("Niveles de accion:", dict(niveles))
    print("Respuestas emitidas:", sum(1 for t in tele if t["respuesta_emitida"]))
    print("Escalados a humano:", sum(1 for t in tele if t["escalado_humano"]))
    print("\n--- EJEMPLOS DE RESPUESTA POR EMOCION/NIVEL ---")
    vistos=set()
    for o in salidas:
        t=o["telemetria"]
        k=(t["emocion_dominante"],t["nivel_accion"])
        if o["respuesta"] and k not in vistos:
            vistos.add(k)
            print(f"\n[{t['emocion_dominante']} | sev={t['severidad']} | {t['nivel_accion']}"
                  + (" | ESCALADO" if t['escalado_humano'] else "") + "]")
            print("  BOT:", o["respuesta"])
