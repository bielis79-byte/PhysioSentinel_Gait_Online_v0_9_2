import json
import math
import os
import re
import shutil
import tempfile
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import requests
import streamlit as st
import toml

APP_VERSION = "0.9.2-online"
TMP_ROOT = Path(tempfile.gettempdir()) / "physiosentinel_gait_online" / "sessions"
TMP_ROOT.mkdir(parents=True, exist_ok=True)

HALPE26 = {
    "Nose": 0,
    "LShoulder": 5, "RShoulder": 6,
    "LElbow": 7, "RElbow": 8,
    "LWrist": 9, "RWrist": 10,
    "LHip": 11, "RHip": 12,
    "LKnee": 13, "RKnee": 14,
    "LAnkle": 15, "RAnkle": 16,
    "Head": 17, "Neck": 18, "Hip": 19,
    "LBigToe": 20, "RBigToe": 21,
    "LSmallToe": 22, "RSmallToe": 23,
    "LHeel": 24, "RHeel": 25,
}
LOWER_BODY = ["LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "LHeel", "RHeel", "LBigToe", "RBigToe"]
FOOT_POINTS = ["LAnkle", "RAnkle", "LHeel", "RHeel", "LBigToe", "RBigToe", "LSmallToe", "RSmallToe"]
UPPER_BODY = ["LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist"]
ASSISTIVE_OPTIONS = ["Sin ayuda", "Bastón", "1 muleta", "2 muletas", "Caminador", "Rollator", "Otra"]
SKELETON = [
    ("LShoulder", "RShoulder"), ("LShoulder", "LHip"), ("RShoulder", "RHip"), ("LHip", "RHip"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"), ("LAnkle", "LHeel"), ("LAnkle", "LBigToe"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"), ("RAnkle", "RHeel"), ("RAnkle", "RBigToe"),
    ("Neck", "Hip"), ("Nose", "Neck"),
]

st.set_page_config(page_title="PhysioSentinel Gait", page_icon="🚶", layout="wide")

# ------------------------- biblioteca de referencias -------------------------
# IMPORTANTE: estas bandas son referencias publicadas/contextuales, no umbrales diagnósticos.
# La comparabilidad depende de edad, sexo, velocidad, protocolo y sistema de medida.
REFERENCE_LIBRARY_VERSION = "2026.09.03"
REFERENCE_LIBRARY = {
    "cadence_exp": {
        "label": "Cadencia",
        "low": 114.95, "high": 118.35, "unit": "pasos/min",
        "population": "Adultos aparentemente sanos, marcha habitual al aire libre",
        "method": "Metaanálisis; media agrupada 116,65 pasos/min (IC95% 114,95–118,35)",
        "source": "Tudor-Locke et al., Sports Medicine 2021",
        "doi": "10.1007/s40279-020-01351-3",
        "applicability": "Contextual. El IC de la media no equivale a un intervalo individual de normalidad. No usar como umbral diagnóstico, especialmente en marcha neurológica."
    },
    "lateral_cadence_exp": {
        "label": "Cadencia lateral",
        "low": 114.95, "high": 118.35, "unit": "pasos/min",
        "population": "Adultos aparentemente sanos, marcha habitual al aire libre",
        "method": "Metaanálisis; media agrupada 116,65 pasos/min (IC95% 114,95–118,35)",
        "source": "Tudor-Locke et al., Sports Medicine 2021",
        "doi": "10.1007/s40279-020-01351-3",
        "applicability": "Contextual. El IC de la media no equivale a un intervalo individual de normalidad. No usar como umbral diagnóstico."
    },
}


# Referencias contextuales adicionales. Se usan solo cuando la definición de la métrica es comparable.
# Para proxies 2D sin equivalencia validada se muestra explícitamente que no existe un rango normativo transferible.
REFERENCE_CONTEXT = {
    "regularity_cv": "Referencia análoga: CV de tiempo de paso ~3.35% como umbral de variabilidad patológica en metaanálisis; no es idéntico a la alternancia 2D de esta app (König et al., 2019, PMID 31639377).",
    "temporal_asymmetry_exp": "Sin umbral universal transferible a este detector 2D. En laboratorio, la simetría del tiempo de apoyo suele ser cercana a 1:1; una cohorte joven femenina reportó ~0.7% a velocidad preferida (PMCID PMC6335661).",
    "stance_asymmetry_2d": "Referencia contextual: el tiempo de apoyo sano es aproximadamente simétrico entre lados; los límites dependen del índice y del sistema de medida. Este valor es una estimación 2D experimental.",
    "stance_pct_2d": "Referencia temporal clásica en marcha adulta confortable: apoyo ~60% del ciclo; contextual, no umbral diagnóstico y dependiente de velocidad/patología.",
    "swing_pct_2d": "Referencia temporal clásica en marcha adulta confortable: oscilación ~40% del ciclo; contextual, no umbral diagnóstico.",
    "double_support_pct_2d": "Referencia temporal clásica en marcha adulta confortable: doble apoyo total ~20–24% del ciclo (dos periodos de ~10–12%); aumenta habitualmente al disminuir la velocidad.",
    "double_support_time_2d": "Sin duración absoluta universal; depende de la duración del ciclo. Referencia porcentual contextual: ~20–24% del ciclo en marcha adulta confortable.",
    "swing_time_l_2d": "Sin duración absoluta universal; referencia proporcional contextual: oscilación ~40% del ciclo en marcha adulta confortable.",
    "swing_time_r_2d": "Sin duración absoluta universal; referencia proporcional contextual: oscilación ~40% del ciclo en marcha adulta confortable.",
    "swing_asymmetry_2d": "Sin umbral 2D universal validado para este detector. En marcha simétrica los tiempos de oscilación deberían aproximarse entre extremidades; interpretar longitudinalmente.",
    "initial_contact_foot_l_deg": "Sin rango normativo 2D universal transferible. La orientación en contacto inicial depende del plano de cámara, velocidad, calzado y estrategia de contacto.",
    "initial_contact_foot_r_deg": "Sin rango normativo 2D universal transferible. La orientación en contacto inicial depende del plano de cámara, velocidad, calzado y estrategia de contacto.",
    "initial_contact_rearfoot_l_deg": "Sin umbral 2D validado universal para inclinación del retropié en contacto inicial; descriptor proyectado.",
    "initial_contact_rearfoot_r_deg": "Sin umbral 2D validado universal para inclinación del retropié en contacto inicial; descriptor proyectado.",
    "loading_knee_l_deg": "Sin umbral 2D universal. Descriptor de estabilidad/alineación de rodilla durante la ventana de respuesta a la carga.",
    "loading_knee_r_deg": "Sin umbral 2D universal. Descriptor de estabilidad/alineación de rodilla durante la ventana de respuesta a la carga.",
    "terminal_foot_l_deg": "Sin rango normativo 2D universal transferible para orientación distal en pre-oscilación/despegue.",
    "terminal_foot_r_deg": "Sin rango normativo 2D universal transferible para orientación distal en pre-oscilación/despegue.",
    "pelvis_obliquity_rom": "No existe un rango normativo directamente transferible al ROM dinámico HALPE26 2D. La oblicuidad pélvica estática en población sana se ha descrito entre 0–5.6°, pero no equivale a este ROM dinámico (PMID 37254005).",
    "shoulder_obliquity_rom": "Sin rango normativo validado para este proxy 2D markerless; priorizar comparación intraindividual.",
    "trunk_lateral_lean_rom": "Sin rango normativo validado para este proxy 2D markerless; depende de velocidad, tarea y estrategia compensatoria.",
    "shoulder_pelvis_rel_rom": "Sin rango normativo validado para este acoplamiento 2D; usar como descriptor longitudinal.",
    "com_lateral_excursion_cm": "Sin rango universal: la excursión lateral del CoM depende de velocidad, ancho de paso y método. Solo se expresa en cm si existe escala espacial calibrada.",
    "bos_width_cm": "Sin rango universal transferible: el ancho de base depende de antropometría, edad y velocidad. Interpretar con contexto y basal individual.",
    "trendelenburg_drop_l_deg": "Sin umbral diagnóstico 2D universal. Descriptor proyectado de caída pélvica durante apoyo monopodal; confirmar clínicamente si es relevante.",
    "trendelenburg_drop_r_deg": "Sin umbral diagnóstico 2D universal. Descriptor proyectado de caída pélvica durante apoyo monopodal; confirmar clínicamente si es relevante.",
    "dynamic_knee_valgus_l_deg": "Sin rango normativo validado para HALPE26 2D. El valgo dinámico es multiplanar; este valor es solo desviación medial proyectada.",
    "dynamic_knee_valgus_r_deg": "Sin rango normativo validado para HALPE26 2D. El valgo dinámico es multiplanar; este valor es solo desviación medial proyectada.",
    "trunk_pelvis_coupling_r": "Sin banda normativa universal para este coeficiente 2D. Valores cercanos a +1 indican acoplamiento en fase y cercanos a -1, contrafase; el significado clínico depende de la tarea.",
    "trunk_pelvis_phase_deg": "Sin banda normativa universal para este desfase 2D. Interpretar longitudinalmente y junto con la calidad del tracking.",
}

def reference_text_for_metric(key):
    base = key
    for pref in ("front_", "lateral_"):
        if base.startswith(pref):
            base = base[len(pref):]
    ref = REFERENCE_LIBRARY.get(key) or REFERENCE_LIBRARY.get(base)
    if ref:
        return f"Referencia poblacional: {ref['low']:.2f}–{ref['high']:.2f} {ref['unit']} ({ref['population']}); contextual, no umbral diagnóstico."
    if base in REFERENCE_CONTEXT:
        return REFERENCE_CONTEXT[base]
    if any(tok in base for tok in ["knee_flex","hip_flex","ankle_angle","shoulder_elev","frontal_knee_dev","foot_progress","rearfoot_tilt","base_width_relative"]):
        return "Sin rango normativo validado directamente transferible a esta métrica 2D proyectada; usar comparación intraindividual y contexto clínico."
    if base in {"tracking_mean","good_frames_pct","foot_visibility_pct","upper_visibility_pct","step_count_consistency_error_pct"}:
        return "Referencia metodológica interna de calidad; no es una variable clínica normativa."
    if base in {"step_events_detected","segment_duration_s","expected_steps_from_cadence","cadence_count_segment","alternation_interval"}:
        return "Sin rango normativo clínico único; variable dependiente de la tarea/segmento y usada para consistencia interna."
    return "Sin rango normativo estandarizado validado para esta definición y método; interpretar respecto al basal individual y al contexto."

REFERENCE_SOURCES = [
    {"Fuente":"Kreusch et al. 2026","Uso":"Mapa de evidencia normativa en adultos sanos 18–65 años (105 estudios; 11.764 participantes)","DOI":"10.1016/j.gaitpost.2026.110176"},
    {"Fuente":"Herssens et al. 2018","Uso":"Parámetros espaciotemporales y variabilidad a lo largo de la vida adulta","DOI":"10.1016/j.gaitpost.2018.06.012"},
    {"Fuente":"Tudor-Locke et al. 2021","Uso":"Velocidad/cadencia de marcha habitual y otros ritmos en adultos sanos","DOI":"10.1007/s40279-020-01351-3"},
    {"Fuente":"Fukuchi et al. 2019","Uso":"Efecto de la velocidad sobre parámetros espaciotemporales, cinemática y cinética","DOI":"10.1186/s12984-019-0559-8"},
    {"Fuente":"Sato et al. 2023","Uso":"Referencias preliminares markerless de tronco y miembro inferior en mayores sanos japoneses","DOI":"10.1298/ptr.E10247"},
]

def reference_for_metric(key):
    return REFERENCE_LIBRARY.get(key)

def reference_position(value, ref):
    if ref is None or value is None or not np.isfinite(float(value)):
        return "Sin referencia"
    v=float(value)
    if v < ref["low"]: return "Por debajo de la banda publicada"
    if v > ref["high"]: return "Por encima de la banda publicada"
    return "Dentro de la banda publicada"

# ------------------------- seguridad / secretos -------------------------
def secret(name, default=None):
    # Streamlit Community Cloud: st.secrets
    try:
        value = st.secrets.get(name, None)
        if value not in (None, ""):
            return value
    except Exception:
        pass

    # Hugging Face Spaces / Docker: secrets como variables de entorno
    return os.getenv(name, default)

SUPABASE_URL = (secret("SUPABASE_URL", "") or "").rstrip("/")
SUPABASE_KEY = secret("SUPABASE_SERVICE_ROLE_KEY", "") or ""
APP_PASSWORD = secret("GAIT_APP_PASSWORD", "") or ""


def require_password():
    if not APP_PASSWORD:
        st.warning("⚠️ GAIT_APP_PASSWORD no está configurada. Modo de prueba sin control de acceso.")
        return True
    if st.session_state.get("authenticated"):
        return True
    st.title("PhysioSentinel Gait")
    st.caption("Acceso protegido")
    pwd = st.text_input("Contraseña", type="password")
    if st.button("Entrar", type="primary"):
        if pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    return False

if not require_password():
    st.stop()

# ------------------------- Supabase REST -------------------------
def sb_ready():
    return bool(SUPABASE_URL and SUPABASE_KEY)


def sb_headers(extra_prefer=None):
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra_prefer:
        h["Prefer"] = extra_prefer
    return h


def sb_request(method, table, params=None, payload=None, prefer=None, timeout=30):
    if not sb_ready():
        raise RuntimeError("Supabase no está configurado en Streamlit Secrets.")
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    r = requests.request(method, url, headers=sb_headers(prefer), params=params, json=payload, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase {r.status_code}: {r.text[:800]}")
    if not r.text.strip():
        return None
    try:
        return r.json()
    except Exception:
        return r.text


def sb_upsert_patient(code):
    data = sb_request(
        "POST", "gait_patients",
        params={"on_conflict": "code"},
        payload={"code": code},
        prefer="resolution=merge-duplicates,return=representation",
    )
    if not data:
        data = sb_request("GET", "gait_patients", params={"code": f"eq.{code}", "select": "id,code"})
    return data[0]["id"]


def sb_create_session(code, record_name, mode, view, meta, assistive_device="Sin ayuda", frontal_orientation="No especificada", meta2=None, calibration_profile_name=None):
    patient_id = sb_upsert_patient(code)
    session_id = str(uuid.uuid4())
    payload = {
        "id": session_id,
        "patient_id": patient_id,
        "record_name": record_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "view": view or "",
        "assistive_device": assistive_device or "Sin ayuda",
        "assisted_gait": bool((assistive_device or "Sin ayuda") != "Sin ayuda"),
        "frontal_orientation": frontal_orientation or "No especificada",
        "fps": float(meta.get("fps", 0)) if meta else None,
        "frames": int(meta.get("frames", 0)) if meta else None,
        "duration_s": float(meta.get("duration", 0)) if meta else None,
        "fps_cam2": float(meta2.get("fps", 0)) if meta2 else None,
        "frames_cam2": int(meta2.get("frames", 0)) if meta2 else None,
        "duration_cam2_s": float(meta2.get("duration", 0)) if meta2 else None,
        "calibration_profile_name": calibration_profile_name or None,
        "video_persisted": False,
        "app_version": APP_VERSION,
    }
    sb_request("POST", "gait_sessions", payload=payload, prefer="return=minimal")
    return session_id


def sb_save_metrics(session_id, metrics, start_s, end_s):
    payload = []
    for m in metrics:
        v = m.get("value")
        if v is not None:
            try:
                v = float(v)
                if not np.isfinite(v):
                    v = None
            except Exception:
                v = None
        payload.append({
            "session_id": session_id,
            "metric_key": m["key"],
            "metric_label": m["label"],
            "value": v,
            "unit": m.get("unit", ""),
            "quality": m.get("quality", ""),
            "notes": m.get("notes", ""),
        })
    sb_request(
        "POST", "gait_metrics",
        params={"on_conflict": "session_id,metric_key"},
        payload=payload,
        prefer="resolution=merge-duplicates,return=minimal",
        timeout=60,
    )
    sb_request(
        "PATCH", "gait_sessions",
        params={"id": f"eq.{session_id}"},
        payload={"segment_start_s": float(start_s), "segment_end_s": float(end_s), "analysis_status": "completed"},
        prefer="return=minimal",
    )


def sb_list_patients():
    if not sb_ready():
        return pd.DataFrame()
    data = sb_request("GET", "gait_patients", params={"select": "id,code,created_at", "order": "code.asc"}) or []
    return pd.DataFrame(data)


def sb_list_calibrations():
    if not sb_ready():
        return []
    try:
        return sb_request(
            "GET", "gait_calibrations",
            params={"select": "id,name,camera_count,notes,created_at", "order": "name.asc"},
        ) or []
    except Exception:
        return []


def sb_get_calibration(name):
    if not sb_ready() or not name:
        return None
    data = sb_request(
        "GET", "gait_calibrations",
        params={"name": f"eq.{name}", "select": "id,name,camera_count,content_toml,notes,created_at", "limit": 1},
    ) or []
    return data[0] if data else None


def sb_upsert_calibration(name, content_toml, notes="", camera_count=2):
    payload = {
        "name": name.strip(),
        "camera_count": int(camera_count),
        "content_toml": content_toml,
        "notes": notes or "",
    }
    data = sb_request(
        "POST", "gait_calibrations",
        params={"on_conflict": "name"},
        payload=payload,
        prefer="resolution=merge-duplicates,return=representation",
    )
    return data[0] if data else None


def sb_update_session_3d(session_id, **fields):
    if not session_id or not fields:
        return
    clean = {k: v for k, v in fields.items() if v is not None}
    if clean:
        sb_request(
            "PATCH", "gait_sessions",
            params={"id": f"eq.{session_id}"},
            payload=clean,
            prefer="return=minimal",
        )


def sb_delete_session(session_id):
    """Elimina de forma permanente un registro y sus métricas asociadas.

    Se borran primero gait_metrics y después gait_sessions para funcionar
    incluso si la FK no tiene ON DELETE CASCADE. No elimina al paciente.
    """
    if not session_id:
        raise ValueError("Falta el identificador de la sesión.")
    sb_request(
        "DELETE", "gait_metrics",
        params={"session_id": f"eq.{session_id}"},
        prefer="return=minimal",
    )
    sb_request(
        "DELETE", "gait_sessions",
        params={"id": f"eq.{session_id}"},
        prefer="return=minimal",
    )
    return True


def sb_patient_history(code):
    if not sb_ready():
        return pd.DataFrame()
    p = sb_request("GET", "gait_patients", params={"code": f"eq.{code}", "select": "id,code", "limit": 1}) or []
    if not p:
        return pd.DataFrame()
    pid = p[0]["id"]
    sessions = sb_request(
        "GET", "gait_sessions",
        params={"patient_id": f"eq.{pid}", "select": "id,created_at,record_name,mode,view,assistive_device,assisted_gait,frontal_orientation,fps,frames,duration_s,fps_cam2,frames_cam2,duration_cam2_s,sync_offset_s,sync_correlation,sync_quality,calibration_profile_name,ready_3d,segment_start_s,segment_end_s,analysis_status", "order": "created_at.asc"},
    ) or []
    if not sessions:
        return pd.DataFrame()
    ids = ",".join(s["id"] for s in sessions)
    metrics = sb_request(
        "GET", "gait_metrics",
        params={"session_id": f"in.({ids})", "select": "session_id,metric_key,metric_label,value,unit,quality,notes"},
    ) or []
    sdf = pd.DataFrame(sessions)
    mdf = pd.DataFrame(metrics)
    if mdf.empty:
        return pd.DataFrame()
    return sdf.merge(mdf, left_on="id", right_on="session_id", how="inner")

# ------------------------- utilidades vídeo -------------------------
def safe_name(text):
    text = (text or "sesion").strip()
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text)[:60] or "sesion"


def video_metadata(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {
        "fps": fps, "frames": frames, "width": width, "height": height,
        "duration": frames / fps if fps > 0 else 0,
        "orientation": "Vertical" if height > width else "Horizontal",
    }


def save_upload(uploaded, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())


def create_temp_session(patient, record):
    old = st.session_state.get("session_dir")
    if old:
        try:
            shutil.rmtree(old, ignore_errors=True)
        except Exception:
            pass
    folder = TMP_ROOT / f"{safe_name(patient)}_{safe_name(record)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    (folder / "videos").mkdir(parents=True, exist_ok=True)
    return folder


def prepare_config(session_dir):
    cfg = {
        "project": {
            "project_dir": str(session_dir),
            "multi_person": False,
            "participant_height": "auto",
            "participant_mass": 70,
            "frame_rate": "auto",
            "frame_range": "auto",
        },
        "pose": {
            "pose_model": "Body_with_feet",
            "mode": "balanced",
            "det_frequency": 4,
            "device": "auto",
            "backend": "auto",
            "display_detection": False,
            "overwrite_pose": True,
            "save_video": "to_video",
            "output_format": "openpose",
            "tracking_mode": "sports2d",
        },
    }
    path = session_dir / "Config.toml"
    with open(path, "w", encoding="utf-8") as f:
        toml.dump(cfg, f)
    return path


def run_pose2sim(config_path):
    from Pose2Sim import Pose2Sim
    Pose2Sim.poseEstimation(str(config_path))


def find_pose_json_dir(session_dir, cam="cam01"):
    pose_dir = session_dir / "pose"
    if not pose_dir.exists():
        return None
    preferred = sorted([p for p in pose_dir.rglob(f"{cam}*_json") if p.is_dir()])
    if preferred:
        return preferred[0]
    candidates = sorted([p for p in pose_dir.rglob("*_json") if p.is_dir()])
    return candidates[0] if candidates else None


def parse_frame_number(path, fallback):
    m = re.search(r"(\d+)(?=\.json$)", path.name)
    return int(m.group(1)) if m else fallback


def load_pose_dataframe(json_dir):
    rows = []
    for i, path in enumerate(sorted(json_dir.glob("*.json"))):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            people = data.get("people", [])
            if not people:
                continue
            pts = people[0].get("pose_keypoints_2d", [])
            if len(pts) < 26 * 3:
                continue
            row = {"frame": parse_frame_number(path, i)}
            for name, idx in HALPE26.items():
                base = idx * 3
                row[f"{name}_x"] = float(pts[base])
                row[f"{name}_y"] = float(pts[base + 1])
                row[f"{name}_score"] = float(pts[base + 2])
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("frame").reset_index(drop=True) if rows else pd.DataFrame()


# ------------------------- v0.9.0 · selección multipersona -------------------------
def _person_points(person):
    pts = person.get("pose_keypoints_2d", []) if isinstance(person, dict) else []
    if len(pts) < 26 * 3:
        return None
    return np.asarray(pts[:26*3], dtype=float).reshape(26, 3)


def _descriptor_from_points(pts, score_thr=0.20):
    if pts is None or pts.shape[0] < 26:
        return None
    valid = np.isfinite(pts[:,0]) & np.isfinite(pts[:,1]) & np.isfinite(pts[:,2]) & (pts[:,2] >= score_thr)
    if int(valid.sum()) < 7:
        return None

    xy = pts[:, :2]
    # Centro robusto: pelvis/hips/shoulders si son visibles; si no, mediana global.
    core_idx = [HALPE26["Hip"], HALPE26["LHip"], HALPE26["RHip"], HALPE26["LShoulder"], HALPE26["RShoulder"]]
    core_valid = [i for i in core_idx if valid[i]]
    if core_valid:
        center = np.nanmedian(xy[core_valid], axis=0)
    else:
        center = np.nanmedian(xy[valid], axis=0)

    xv, yv = xy[valid,0], xy[valid,1]
    x1, x2 = float(np.nanmin(xv)), float(np.nanmax(xv))
    y1, y2 = float(np.nanmin(yv)), float(np.nanmax(yv))
    w, h = max(1.0, x2-x1), max(1.0, y2-y1)
    scale = float(max(h, w, np.hypot(w, h) * 0.70))
    shape = np.full((26,2), np.nan, dtype=float)
    shape[valid] = (xy[valid] - center) / max(scale, 1.0)

    return {
        "center": np.asarray(center, dtype=float),
        "bbox": np.asarray([x1,y1,x2,y2], dtype=float),
        "scale": scale,
        "shape": shape,
        "valid": valid,
        "mean_score": float(np.nanmean(pts[valid,2])),
        "n_valid": int(valid.sum()),
    }


def _bbox_iou(a, b):
    if a is None or b is None:
        return 0.0
    x1=max(float(a[0]),float(b[0])); y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2])); y2=min(float(a[3]),float(b[3]))
    iw=max(0.0,x2-x1); ih=max(0.0,y2-y1)
    inter=iw*ih
    aa=max(0.0,float(a[2]-a[0]))*max(0.0,float(a[3]-a[1]))
    bb=max(0.0,float(b[2]-b[0]))*max(0.0,float(b[3]-b[1]))
    den=aa+bb-inter
    return float(inter/den) if den>0 else 0.0


def _shape_distance(a, b):
    va = a.get("shape"); vb = b.get("shape")
    if va is None or vb is None:
        return 1.0
    ok = np.isfinite(va[:,0]) & np.isfinite(va[:,1]) & np.isfinite(vb[:,0]) & np.isfinite(vb[:,1])
    if int(ok.sum()) < 5:
        return 1.0
    d = np.linalg.norm(va[ok]-vb[ok], axis=1)
    return float(np.nanmedian(d))


def _identity_cost(prev, cand, predicted_center=None):
    if prev is None or cand is None:
        return np.inf
    pc = np.asarray(predicted_center if predicted_center is not None else prev["center"], dtype=float)
    scale = max(float(prev.get("scale",1.0)), float(cand.get("scale",1.0)), 1.0)
    center_d = float(np.linalg.norm(np.asarray(cand["center"])-pc) / scale)
    iou_pen = 1.0 - _bbox_iou(prev.get("bbox"), cand.get("bbox"))
    scale_pen = abs(float(np.log(max(cand.get("scale",1.0),1.0) / max(prev.get("scale",1.0),1.0))))
    shape_pen = min(2.0, _shape_distance(prev, cand))
    conf_pen = max(0.0, 0.55 - float(cand.get("mean_score",0.0)))
    return float(0.42*center_d + 0.23*iou_pen + 0.14*scale_pen + 0.17*shape_pen + 0.04*conf_pen)


def _frame_people(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    out=[]
    for idx, person in enumerate(data.get("people", []) or []):
        pts=_person_points(person)
        desc=_descriptor_from_points(pts)
        if desc is None:
            continue
        out.append({"person_index":int(idx), "pts":pts, "desc":desc})
    return out


def scan_subject_candidates(json_dir):
    """
    Escoge automáticamente UN FOTOGRAMA de selección con la mayor cantidad de
    sujetos suficientemente visibles. No elige al paciente: solo prepara las
    opciones para que el clínico lo seleccione manualmente.
    """
    paths=sorted(Path(json_dir).glob("*.json"))
    if not paths:
        return None

    best=None
    # Preferir la primera mitad del ensayo para que la elección se haga antes
    # de cruces/giro, pero permitir cualquier frame si allí hay mejor visibilidad.
    scored=[]
    for i,p in enumerate(paths):
        people=_frame_people(p)
        if not people:
            continue
        n=len(people)
        quality=float(np.mean([x["desc"]["mean_score"] for x in people]))
        central_bonus=0.05 if i <= max(1, int(0.55*len(paths))) else 0.0
        score=n*10.0 + quality + central_bonus
        scored.append((score, -i, p, people))
    if not scored:
        return None
    scored.sort(key=lambda x:(x[0],x[1]), reverse=True)
    _,_,p,people=scored[0]
    frame_no=parse_frame_number(p,0)

    # Etiquetas estables para la UI: de izquierda a derecha en el frame elegido.
    ordered=sorted(people, key=lambda x: float(x["desc"]["center"][0]))
    candidates=[]
    for pos, item in enumerate(ordered, start=1):
        d=item["desc"]
        candidates.append({
            "label":f"Sujeto {pos}",
            "person_index":int(item["person_index"]),
            "center_x":float(d["center"][0]),
            "center_y":float(d["center"][1]),
            "bbox":[float(v) for v in d["bbox"]],
            "mean_score":float(d["mean_score"]),
            "n_valid":int(d["n_valid"]),
        })
    return {"frame":int(frame_no), "candidates":candidates, "max_people":int(len(candidates))}


def render_subject_preview(video_path, selection):
    if not selection:
        return None
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_no=int(selection["frame"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame=cap.read()
    cap.release()
    if not ok or frame is None:
        return None

    for cand in selection["candidates"]:
        x1,y1,x2,y2=[int(round(v)) for v in cand["bbox"]]
        x1=max(0,x1); y1=max(0,y1); x2=min(frame.shape[1]-1,x2); y2=min(frame.shape[0]-1,y2)
        # Sin colores clínicamente semánticos: etiquetas y cajas de alto contraste.
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),3)
        cv2.putText(frame,cand["label"],(x1,max(24,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),3,cv2.LINE_AA)
        cv2.putText(frame,cand["label"],(x1,max(24,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),1,cv2.LINE_AA)
    ok, enc=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),90])
    return enc.tobytes() if ok else None


def _row_from_tracked(frame_no, pts):
    row={"frame":int(frame_no)}
    for name,idx in HALPE26.items():
        row[f"{name}_x"]=float(pts[idx,0])
        row[f"{name}_y"]=float(pts[idx,1])
        row[f"{name}_score"]=float(pts[idx,2])
    return row


def load_pose_dataframe_tracked(json_dir, anchor_frame, selected_person_index):
    """
    Seguimiento de identidad bloqueado desde el sujeto elegido manualmente.

    - Sigue hacia delante y hacia atrás desde el frame de selección.
    - Combina continuidad espacial, tamaño corporal, IoU y forma esquelética.
    - Si dos candidatos son demasiado parecidos o el coste es excesivo,
      NO cambia silenciosamente de persona: excluye ese frame.
    """
    paths=sorted(Path(json_dir).glob("*.json"))
    if not paths:
        return pd.DataFrame(), {}

    frames=[]
    for i,p in enumerate(paths):
        fn=parse_frame_number(p,i)
        frames.append((int(fn),p))
    frame_to_pos={fn:i for i,(fn,_) in enumerate(frames)}
    if int(anchor_frame) not in frame_to_pos:
        # frame más cercano, por robustez ante nombres de archivo no contiguos
        anchor_frame=min(frame_to_pos, key=lambda x:abs(x-int(anchor_frame)))
    a_pos=frame_to_pos[int(anchor_frame)]

    anchor_people=_frame_people(frames[a_pos][1])
    anchor=next((x for x in anchor_people if int(x["person_index"])==int(selected_person_index)),None)
    if anchor is None:
        return pd.DataFrame(), {"quality":"No fiable","reason":"No se encontró el sujeto seleccionado en el frame ancla."}

    accepted={int(anchor_frame):anchor}
    ambiguous=set()
    missing=set()
    switches_prevented=0
    max_people=0

    def walk(indices):
        nonlocal switches_prevented,max_people
        prev=anchor
        prev2=None
        gap=0
        for pos in indices:
            fn,p=frames[pos]
            people=_frame_people(p)
            max_people=max(max_people,len(people))
            if not people:
                missing.add(fn); gap+=1
                continue

            pred=None
            if prev2 is not None and gap==0:
                v=np.asarray(prev["desc"]["center"])-np.asarray(prev2["desc"]["center"])
                pred=np.asarray(prev["desc"]["center"])+v

            scored=[]
            for cand in people:
                cost=_identity_cost(prev["desc"],cand["desc"],predicted_center=pred)
                scored.append((cost,cand))
            scored.sort(key=lambda x:x[0])
            best_cost,best=scored[0]
            second_cost=scored[1][0] if len(scored)>1 else np.inf
            margin=second_cost-best_cost

            # Umbral algo más permisivo tras una oclusión breve, pero nunca
            # permite un salto grande de identidad.
            max_cost=0.92 if gap==0 else min(1.15,0.92+0.025*min(gap,9))
            ambiguous_match = (
                best_cost > max_cost or
                (len(scored)>1 and margin < 0.11 and best_cost > 0.26)
            )
            if ambiguous_match:
                ambiguous.add(fn)
                switches_prevented += 1 if len(scored)>1 else 0
                gap += 1
                # conservar identidad previa como referencia; no adoptar candidato dudoso
                continue

            accepted[fn]=best
            prev2=prev
            prev=best
            gap=0

    walk(range(a_pos+1,len(frames)))
    # Reiniciar referencia para seguimiento hacia atrás.
    prev_anchor=anchor
    # La función walk mantiene variables locales prev, por lo que basta otra llamada
    walk(range(a_pos-1,-1,-1))

    rows=[]
    for fn,_ in frames:
        item=accepted.get(fn)
        if item is not None:
            rows.append(_row_from_tracked(fn,item["pts"]))

    df=pd.DataFrame(rows).sort_values("frame").reset_index(drop=True) if rows else pd.DataFrame()
    total=len(frames)
    reliable=len(rows)
    excluded=max(0,total-reliable)
    continuity=100.0*reliable/total if total else np.nan
    ambiguous_pct=100.0*len(ambiguous)/total if total else np.nan
    missing_pct=100.0*len(missing)/total if total else np.nan

    if np.isfinite(continuity):
        quality="Alta" if continuity>=90 else ("Moderada" if continuity>=75 else "Baja")
    else:
        quality="No fiable"
    info={
        "manual":True,
        "anchor_frame":int(anchor_frame),
        "selected_person_index":int(selected_person_index),
        "identity_continuity_pct":float(continuity) if np.isfinite(continuity) else np.nan,
        "ambiguous_excluded_pct":float(ambiguous_pct) if np.isfinite(ambiguous_pct) else np.nan,
        "missing_pose_pct":float(missing_pct) if np.isfinite(missing_pct) else np.nan,
        "frames_total":int(total),
        "frames_reliable":int(reliable),
        "frames_excluded":int(excluded),
        "ambiguous_frames":int(len(ambiguous)),
        "switches_prevented":int(switches_prevented),
        "max_people_detected":int(max_people),
        "quality":quality,
    }
    return df,info


def tracking_metrics(info, prefix="", view_label=""):
    if not info or not info.get("manual"):
        return []
    p=(prefix+"_" if prefix else "")
    view_note=f" ({view_label})" if view_label else ""
    return [
        {"key":p+"subject_manual_selection_flag","label":"Selección manual de sujeto"+view_note,"value":1.0,"unit":"bool","quality":"Bloqueado","notes":"v0.9.1: el paciente fue seleccionado explícitamente por el clínico antes del análisis biomecánico."},
        {"key":p+"identity_continuity_pct","label":"Continuidad de identidad"+view_note,"value":info.get("identity_continuity_pct"),"unit":"%","quality":info.get("quality","No fiable"),"notes":"Porcentaje de frames en los que la identidad seleccionada se mantuvo con confianza suficiente."},
        {"key":p+"identity_ambiguous_excluded_pct","label":"Frames ambiguos excluidos"+view_note,"value":info.get("ambiguous_excluded_pct"),"unit":"%","quality":"Control interno","notes":"Frames descartados por riesgo de confundir al paciente con otra persona."},
        {"key":p+"identity_frames_excluded","label":"Frames excluidos por identidad"+view_note,"value":info.get("frames_excluded"),"unit":"frames","quality":"Control interno","notes":"Incluye oclusiones, pérdidas de pose y emparejamientos ambiguos; nunca se sustituyen silenciosamente por otro sujeto."},
        {"key":p+"max_people_detected","label":"Máximo de sujetos detectados"+view_note,"value":info.get("max_people_detected"),"unit":"personas","quality":"Directa","notes":"Máximo número de personas con pose utilizable observado durante el seguimiento."},
    ]


def point_angle(ax, ay, bx, by, cx, cy):
    ba = np.array([ax - bx, ay - by], dtype=float)
    bc = np.array([cx - bx, cy - by], dtype=float)
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba == 0 or nbc == 0:
        return np.nan
    c = np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def robust_rom(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 95) - np.percentile(x, 5)) if len(x) >= 5 else np.nan


def rolling_smooth(arr, window=7):
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().to_numpy()


def zero_crossings(signal):
    s = np.asarray(signal, dtype=float)
    out = []
    for i in range(1, len(s)):
        if not (np.isfinite(s[i-1]) and np.isfinite(s[i])):
            continue
        if (s[i-1] <= 0 < s[i]) or (s[i-1] >= 0 > s[i]):
            out.append(i)
    return np.asarray(out, dtype=int)


def quality_label(score):
    return "Alta" if score >= 0.80 else ("Moderada" if score >= 0.65 else "Baja")


def add_angle_columns(seg):
    seg = seg.copy()
    for side in ("L", "R"):
        knee, hip, ankle, shoulder = [], [], [], []
        for _, r in seg.iterrows():
            ka = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"])
            ha = point_angle(r[f"{side}Shoulder_x"], r[f"{side}Shoulder_y"], r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"])
            aa = point_angle(r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"], r[f"{side}BigToe_x"], r[f"{side}BigToe_y"])
            sa = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Shoulder_x"], r[f"{side}Shoulder_y"], r[f"{side}Elbow_x"], r[f"{side}Elbow_y"])
            knee.append(180.0 - ka if np.isfinite(ka) else np.nan)
            hip.append(180.0 - ha if np.isfinite(ha) else np.nan)
            ankle.append(aa)
            shoulder.append(sa)
        seg[f"{side}_knee_flex"] = knee
        seg[f"{side}_hip_flex"] = hip
        seg[f"{side}_ankle_angle"] = ankle
        seg[f"{side}_shoulder_elev"] = shoulder
    return seg


def axis_angle_to_vertical(x1, y1, x2, y2):
    """Ángulo firmado de un eje 2D respecto a la vertical de la imagen, plegado a [-90, 90]."""
    dx, dy = float(x2-x1), float(y2-y1)
    if not np.isfinite(dx) or not np.isfinite(dy) or (abs(dx)+abs(dy) == 0):
        return np.nan
    a = float(np.degrees(np.arctan2(dx, -dy)))
    while a > 90: a -= 180
    while a < -90: a += 180
    return a


def axis_angle_to_horizontal(x1, y1, x2, y2):
    dx, dy = float(x2-x1), float(y2-y1)
    if not np.isfinite(dx) or not np.isfinite(dy) or (abs(dx)+abs(dy) == 0):
        return np.nan
    a = float(np.degrees(np.arctan2(dy, dx)))
    while a > 90: a -= 180
    while a < -90: a += 180
    return a


def add_frontal_columns(seg):
    """Métricas proyectadas para vista frontal/posterior. No equivalen a rotaciones 3D ni a pronación clínica."""
    seg = seg.copy()
    for side in ("L", "R"):
        knee_dev, foot_prog, rearfoot = [], [], []
        for _, r in seg.iterrows():
            ka = point_angle(r[f"{side}Hip_x"], r[f"{side}Hip_y"], r[f"{side}Knee_x"], r[f"{side}Knee_y"], r[f"{side}Ankle_x"], r[f"{side}Ankle_y"])
            knee_dev.append(abs(180.0-ka) if np.isfinite(ka) else np.nan)
            foot_prog.append(axis_angle_to_vertical(r[f"{side}Heel_x"], r[f"{side}Heel_y"], r[f"{side}BigToe_x"], r[f"{side}BigToe_y"]))
            rearfoot.append(axis_angle_to_vertical(r[f"{side}Ankle_x"], r[f"{side}Ankle_y"], r[f"{side}Heel_x"], r[f"{side}Heel_y"]))
        seg[f"{side}_frontal_knee_dev"] = knee_dev
        seg[f"{side}_foot_progress_proj"] = foot_prog
        seg[f"{side}_rearfoot_tilt_proj"] = rearfoot
    seg["pelvis_obliquity"] = [axis_angle_to_horizontal(r.LHip_x, r.LHip_y, r.RHip_x, r.RHip_y) for _, r in seg.iterrows()]
    seg["shoulder_obliquity"] = [axis_angle_to_horizontal(r.LShoulder_x, r.LShoulder_y, r.RShoulder_x, r.RShoulder_y) for _, r in seg.iterrows()]
    # Tronco en plano frontal/posterior: eje entre el centro pélvico y el centro de hombros.
    trunk_lean = []
    for _, r in seg.iterrows():
        hip_x = (r.LHip_x + r.RHip_x) / 2.0
        hip_y = (r.LHip_y + r.RHip_y) / 2.0
        sh_x = (r.LShoulder_x + r.RShoulder_x) / 2.0
        sh_y = (r.LShoulder_y + r.RShoulder_y) / 2.0
        trunk_lean.append(axis_angle_to_vertical(hip_x, hip_y, sh_x, sh_y))
    seg["trunk_lateral_lean"] = trunk_lean
    seg["shoulder_pelvis_rel"] = seg["shoulder_obliquity"] - seg["pelvis_obliquity"]
    pelvis_w = np.abs(seg.RHip_x.to_numpy(float) - seg.LHip_x.to_numpy(float))
    ankle_w = np.abs(seg.RAnkle_x.to_numpy(float) - seg.LAnkle_x.to_numpy(float))
    seg["base_width_relative"] = np.divide(ankle_w, pelvis_w, out=np.full_like(ankle_w, np.nan), where=pelvis_w>1e-6)
    return seg


def _foot_centroid(seg, side):
    x = (seg[f"{side}Ankle_x"].to_numpy(float) + seg[f"{side}Heel_x"].to_numpy(float) + seg[f"{side}BigToe_x"].to_numpy(float)) / 3.0
    y = (seg[f"{side}Ankle_y"].to_numpy(float) + seg[f"{side}Heel_y"].to_numpy(float) + seg[f"{side}BigToe_y"].to_numpy(float)) / 3.0
    return x, y


def _bool_runs(mask):
    """Devuelve runs True como (inicio, fin_exclusivo)."""
    m = np.asarray(mask, dtype=bool)
    runs = []
    start = None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        if start is not None and ((not v) or i == len(m) - 1):
            end = i if not v else i + 1
            runs.append((int(start), int(end)))
            start = None
    return runs


def _fill_short_false_gaps(mask, max_gap_frames, speed=None, speed_ceiling=None):
    m = np.asarray(mask, dtype=bool).copy()
    if len(m) == 0 or max_gap_frames <= 0:
        return m
    for a, b in _bool_runs(~m):
        if a == 0 or b == len(m):
            continue
        if (b - a) <= max_gap_frames:
            if speed is None or speed_ceiling is None:
                m[a:b] = True
            else:
                g = np.asarray(speed[a:b], float)
                if len(g) and np.isfinite(g).any() and float(np.nanmedian(g)) <= float(speed_ceiling):
                    m[a:b] = True
    return m


def _remove_short_true_runs(mask, min_run_frames):
    m = np.asarray(mask, dtype=bool).copy()
    for a, b in _bool_runs(m):
        if (b - a) < min_run_frames:
            m[a:b] = False
    return m


def _support_mask_2d(seg, side, fps, expected_stride_s=np.nan):
    """
    v0.9.2: estimación cinemática 2D de apoyo con continuidad temporal.
    Evita fragmentar un apoyo real en microbloques por jitter del tracking.

    Sigue siendo markerless 2D: NO sustituye footswitch, plataforma de fuerzas
    ni presión plantar.
    """
    x, y = _foot_centroid(seg, side)
    xs = (
        pd.Series(x).interpolate(limit_direction="both")
        .rolling(5, center=True, min_periods=1).median()
        .rolling(5, center=True, min_periods=1).mean()
        .to_numpy(float)
    )
    ys = (
        pd.Series(y).interpolate(limit_direction="both")
        .rolling(5, center=True, min_periods=1).median()
        .rolling(5, center=True, min_periods=1).mean()
        .to_numpy(float)
    )
    speed_px_s = np.hypot(np.gradient(xs), np.gradient(ys)) * float(fps)

    hip_x = (seg.LHip_x.to_numpy(float) + seg.RHip_x.to_numpy(float)) / 2.0
    hip_y = (seg.LHip_y.to_numpy(float) + seg.RHip_y.to_numpy(float)) / 2.0
    sh_x = (seg.LShoulder_x.to_numpy(float) + seg.RShoulder_x.to_numpy(float)) / 2.0
    sh_y = (seg.LShoulder_y.to_numpy(float) + seg.RShoulder_y.to_numpy(float)) / 2.0
    torso = np.hypot(sh_x - hip_x, sh_y - hip_y)
    torso = pd.Series(torso).replace([np.inf, -np.inf], np.nan).interpolate(limit_direction="both").to_numpy(float)
    med_torso = float(np.nanmedian(torso)) if np.isfinite(torso).any() else np.nan
    if not np.isfinite(med_torso) or med_torso < 5:
        med_torso = 1.0
    speed = speed_px_s / med_torso

    finite = speed[np.isfinite(speed)]
    if len(finite) < max(20, int(fps)):
        return np.zeros(len(seg), dtype=bool), speed, {
            "quality": "No fiable", "score": 0.0, "reason": "Señal de pie insuficiente."
        }

    low = float(np.nanpercentile(finite, 38))
    high = float(np.nanpercentile(finite, 72))
    if high <= low:
        high = low + max(float(np.nanstd(finite)) * 0.35, 1e-6)

    state_support = bool(speed[0] <= (low + high) / 2.0) if np.isfinite(speed[0]) else True
    mask = np.zeros(len(speed), dtype=bool)
    for i, s in enumerate(speed):
        if not np.isfinite(s):
            mask[i] = state_support
            continue
        if state_support and s > high:
            state_support = False
        elif (not state_support) and s < low:
            state_support = True
        mask[i] = state_support

    if np.isfinite(expected_stride_s) and expected_stride_s > 0:
        gap_s = min(0.40, max(0.12, 0.09 * float(expected_stride_s)))
        min_support_s = min(0.80, max(0.18, 0.14 * float(expected_stride_s)))
        min_swing_s = min(0.50, max(0.12, 0.07 * float(expected_stride_s)))
    else:
        gap_s, min_support_s, min_swing_s = 0.20, 0.20, 0.12

    mask = _fill_short_false_gaps(
        mask,
        max_gap_frames=max(1, int(round(gap_s * fps))),
        speed=speed,
        speed_ceiling=high * 1.10,
    )
    mask = _remove_short_true_runs(mask, max(1, int(round(min_support_s * fps))))

    swing = ~mask
    swing = _remove_short_true_runs(swing, max(1, int(round(min_swing_s * fps))))
    mask = ~swing
    mask = _remove_short_true_runs(mask, max(1, int(round(min_support_s * fps))))

    return mask.astype(bool), speed, {
        "quality": "Pendiente", "score": np.nan,
        "reason": "Máscara estabilizada por histéresis y continuidad temporal.",
        "low_thr": low, "high_thr": high,
    }


def _support_cycle_summary(mask, fps, expected_stride_s=np.nan):
    """
    Convierte una máscara continua en ciclos IC→TO→siguiente IC y puntúa
    su consistencia. Los límites son controles matemáticos amplios, no
    rangos clínicos de normalidad.
    """
    starts, ends = _edges(mask)
    stance, swing, cycles, stance_pct = [], [], [], []
    valid_ic, valid_to, valid_next_ic = [], [], []

    for j in range(len(starts) - 1):
        ic = int(starts[j])
        nxt = int(starts[j + 1])
        offs = ends[(ends > ic) & (ends <= nxt)]
        if not len(offs):
            continue
        to = int(offs[0])
        cyc = (nxt - ic) / float(fps)
        st = (to - ic) / float(fps)
        sw = (nxt - to) / float(fps)
        if cyc <= 0 or st <= 0 or sw < 0:
            continue
        ratio = st / cyc
        if ratio < 0.10 or ratio > 0.985:
            continue
        if np.isfinite(expected_stride_s) and expected_stride_s > 0:
            if cyc < 0.45 * expected_stride_s or cyc > 1.80 * expected_stride_s:
                continue

        stance.append(st)
        swing.append(sw)
        cycles.append(cyc)
        stance_pct.append(ratio * 100.0)
        valid_ic.append(ic)
        valid_to.append(to)
        valid_next_ic.append(nxt)

    stance = np.asarray(stance, float)
    swing = np.asarray(swing, float)
    cycles = np.asarray(cycles, float)
    stance_pct = np.asarray(stance_pct, float)

    score = 100.0
    reasons = []

    if len(cycles) < 2:
        score -= 55
        reasons.append("menos de 2 ciclos completos")
    elif len(cycles) < 3:
        score -= 15
        reasons.append("pocos ciclos completos")

    err = np.nan
    if len(cycles) and np.isfinite(expected_stride_s) and expected_stride_s > 0:
        cyc_med = float(np.nanmedian(cycles))
        err = abs(cyc_med - expected_stride_s) / expected_stride_s * 100.0
        if err > 50:
            score -= 45
            reasons.append("periodo de ciclo discordante con la cadencia")
        elif err > 30:
            score -= 25
            reasons.append("periodo de ciclo moderadamente discordante")
        elif err > 18:
            score -= 10
            reasons.append("periodo de ciclo con ligera discordancia")

    fragmentation = np.nan
    if np.isfinite(expected_stride_s) and expected_stride_s > 0 and len(mask) > 0:
        duration = len(mask) / float(fps)
        expected_same_side_cycles = max(duration / expected_stride_s, 1.0)
        fragmentation = len(starts) / expected_same_side_cycles
        if fragmentation > 2.2:
            score -= 35
            reasons.append("segmentación fragmentada")
        elif fragmentation > 1.6:
            score -= 15
            reasons.append("posible fragmentación residual")

    score = float(max(0.0, min(100.0, score)))
    quality = "Alta" if score >= 75 else ("Moderada" if score >= 55 else "No fiable")

    return {
        "stance": stance,
        "swing": swing,
        "cycle": cycles,
        "stance_pct": stance_pct,
        "swing_pct": 100.0 - stance_pct if len(stance_pct) else np.asarray([], float),
        "ic": np.asarray(valid_ic, int),
        "to": np.asarray(valid_to, int),
        "next_ic": np.asarray(valid_next_ic, int),
        "starts": np.asarray(starts, int),
        "ends": np.asarray(ends, int),
        "quality": quality,
        "score": score,
        "cycle_error_pct": float(err) if np.isfinite(err) else np.nan,
        "fragmentation_index": float(fragmentation) if np.isfinite(fragmentation) else np.nan,
        "reason": ", ".join(reasons) if reasons else "consistencia temporal suficiente",
    }



def _automatic_straight_walking_mask(seg, fps):
    """
    v0.9.2: detecta y excluye automáticamente transiciones/giro de 180° en
    registros frontal/posterior.

    Combina dos señales normalizadas:
      1) estrechamiento transitorio del ancho de hombros/caderas respecto al
         tronco (el cuerpo se pone de perfil durante el giro);
      2) inversión sostenida de la tendencia de tamaño corporal aparente
         (alejarse -> acercarse, o viceversa).

    Es un filtro de calidad, no un detector clínico de giro.
    Si la evidencia no es suficiente, solo recorta bordes del registro.
    """
    n = len(seg)
    if n == 0:
        return np.zeros(0, dtype=bool), {
            "turn_detected": False, "excluded_pct": np.nan,
            "reason": "segmento vacío"
        }

    fps = float(max(fps, 1.0))
    edge = min(max(1, int(round(0.35 * fps))), max(1, n // 8))
    valid = np.ones(n, dtype=bool)
    if n > 2 * edge:
        valid[:edge] = False
        valid[-edge:] = False

    lx = seg.LShoulder_x.to_numpy(float); rx = seg.RShoulder_x.to_numpy(float)
    ly = seg.LShoulder_y.to_numpy(float); ry = seg.RShoulder_y.to_numpy(float)
    lhx = seg.LHip_x.to_numpy(float); rhx = seg.RHip_x.to_numpy(float)
    lhy = seg.LHip_y.to_numpy(float); rhy = seg.RHip_y.to_numpy(float)

    shoulder_w = np.abs(rx - lx)
    hip_w = np.abs(rhx - lhx)
    shoulder_y = (ly + ry) / 2.0
    hip_y = (lhy + rhy) / 2.0
    torso_h = np.abs(hip_y - shoulder_y)

    def _smooth(a, w):
        return (
            pd.Series(a).replace([np.inf, -np.inf], np.nan)
            .interpolate(limit_direction="both")
            .rolling(w, center=True, min_periods=1).median()
            .rolling(w, center=True, min_periods=1).mean()
            .to_numpy(float)
        )

    w = max(5, int(round(0.25 * fps)) | 1)
    shoulder_w = _smooth(shoulder_w, w)
    hip_w = _smooth(hip_w, w)
    torso_h = _smooth(torso_h, w)

    denom = np.maximum(torso_h, 1e-6)
    frontal_ratio = 0.65 * (shoulder_w / denom) + 0.35 * (hip_w / denom)
    finite_ratio = frontal_ratio[np.isfinite(frontal_ratio)]
    turn_candidates = np.zeros(n, dtype=bool)

    # Evidencia 1: orientación transitoria de perfil.
    if len(finite_ratio) >= max(20, int(fps)):
        reference = float(np.nanpercentile(finite_ratio, 75))
        if np.isfinite(reference) and reference > 1e-6:
            turn_candidates |= frontal_ratio < (0.62 * reference)

    # Evidencia 2: inversión sostenida del tamaño aparente del cuerpo.
    # Evita declarar giro por un único frame: compara tendencias a ambos lados.
    scale = _smooth(torso_h, max(7, int(round(0.40 * fps)) | 1))
    look = max(5, int(round(0.85 * fps)))
    reversal_scores = np.zeros(n, dtype=float)
    for i in range(look, n - look):
        pre = scale[i] - scale[i - look]
        post = scale[i + look] - scale[i]
        local = float(np.nanmedian(scale[max(0, i-look):min(n, i+look+1)]))
        if not np.isfinite(local) or local <= 1e-6:
            continue
        # signo opuesto y cambio conjunto de al menos ~5 % del tamaño local
        if np.isfinite(pre) and np.isfinite(post) and pre * post < 0:
            reversal_scores[i] = (abs(pre) + abs(post)) / local

    if np.nanmax(reversal_scores) >= 0.05:
        i0 = int(np.nanargmax(reversal_scores))
        turn_candidates[i0] = True

    # Agrupa el giro y añade margen temporal para no contaminar ciclos vecinos.
    turn_detected = bool(turn_candidates.any())
    if turn_detected:
        idx = np.where(turn_candidates)[0]
        # Si hay muchos candidatos, conservar el cluster principal alrededor
        # del punto con menor frontalidad / mayor inversión.
        center = int(np.median(idx))
        if np.nanmax(reversal_scores) >= 0.05:
            center = int(np.nanargmax(reversal_scores))
        pad = max(1, int(round(0.75 * fps)))
        a = max(0, center - pad)
        b = min(n, center + pad + 1)
        valid[a:b] = False

    # No aplicar una exclusión automática agresiva si deja poco material útil.
    min_keep = max(int(round(3.0 * fps)), int(round(0.45 * n)))
    if valid.sum() < min_keep:
        valid[:] = True
        if n > 2 * edge:
            valid[:edge] = False
            valid[-edge:] = False
        turn_detected = False
        reason = "evidencia de giro no suficientemente robusta; solo se recortan bordes"
    else:
        reason = "giro/transición excluido automáticamente" if turn_detected else "sin giro robusto detectado; bordes excluidos"

    return valid, {
        "turn_detected": turn_detected,
        "excluded_pct": float((1.0 - valid.mean()) * 100.0),
        "usable_pct": float(valid.mean() * 100.0),
        "reason": reason,
    }


def _filter_support_summary_to_mask(summary, valid_mask, fps, min_fraction=0.90):
    """
    Conserva únicamente ciclos IC→TO→siguiente IC contenidos casi por completo
    en el dominio rectilíneo válido.
    """
    valid_mask = np.asarray(valid_mask, dtype=bool)
    keys = ["stance", "swing", "cycle", "stance_pct", "swing_pct", "ic", "to", "next_ic"]
    arrs = {k: np.asarray(summary.get(k, [])) for k in keys}

    ncyc = len(arrs["ic"])
    keep = []
    for j in range(ncyc):
        ic = int(arrs["ic"][j])
        nxt = int(arrs["next_ic"][j]) if j < len(arrs["next_ic"]) else int(round(ic + arrs["cycle"][j] * fps))
        a, b = max(0, ic), min(len(valid_mask), max(ic + 1, nxt))
        frac = float(valid_mask[a:b].mean()) if b > a else 0.0
        keep.append(frac >= float(min_fraction))
    keep = np.asarray(keep, dtype=bool)

    out = dict(summary)
    for k in keys:
        if len(arrs[k]) == ncyc:
            out[k] = arrs[k][keep]

    # Cobertura temporal de ciclos aceptados.
    coverage = np.zeros(len(valid_mask), dtype=bool)
    for ic, nxt in zip(np.asarray(out.get("ic", []), int), np.asarray(out.get("next_ic", []), int)):
        coverage[max(0, ic):min(len(coverage), max(ic + 1, nxt))] = True
    out["coverage_mask"] = coverage & valid_mask

    # Reevalúa la calidad si la exclusión deja muy pocos ciclos.
    n = len(out.get("cycle", []))
    score = float(summary.get("score", 0.0))
    reasons = [str(summary.get("reason", ""))]
    if n < 2:
        score = min(score, 45.0)
        reasons.append("menos de 2 ciclos rectilíneos válidos")
    elif n < 3:
        score = min(score, 70.0)
        reasons.append("solo 2 ciclos rectilíneos válidos")
    quality = "Alta" if score >= 75 else ("Moderada" if score >= 55 else "No fiable")
    out["score"] = score
    out["quality"] = quality
    out["reason"] = ", ".join([r for r in reasons if r])
    return out


def _contact_step_metrics(L, R, fps):
    """
    Cadencia, CV y asimetría a partir de contactos iniciales I/D de ciclos
    validados. Evita usar cruces geométricos pares/impares como sustituto de lado.
    """
    events = []
    for i in np.asarray(L.get("ic", []), int):
        events.append((int(i), "L"))
    for i in np.asarray(R.get("ic", []), int):
        events.append((int(i), "R"))
    events.sort(key=lambda z: z[0])

    # Fusiona duplicados muy próximos del mismo evento y exige alternancia I/D.
    cleaned = []
    min_sep = max(1, int(round(0.18 * float(fps))))
    for idx, side in events:
        if cleaned and idx - cleaned[-1][0] < min_sep:
            # En caso de conflicto conservar el evento que mantiene alternancia.
            if side != cleaned[-1][1] and len(cleaned) >= 2 and side != cleaned[-2][1]:
                cleaned[-1] = (idx, side)
            continue
        if cleaned and side == cleaned[-1][1]:
            continue
        cleaned.append((idx, side))

    if len(cleaned) < 4:
        return {
            "cadence": np.nan, "mean_step": np.nan, "cv": np.nan, "asym": np.nan,
            "events": cleaned, "n_intervals": 0, "quality": "No fiable",
            "reason": "menos de 4 contactos alternantes válidos"
        }

    frames = np.asarray([e[0] for e in cleaned], float)
    sides = [e[1] for e in cleaned]
    intervals = np.diff(frames) / float(fps)
    med = float(np.nanmedian(intervals))

    # Solo elimina intervalos manifiestamente incompatibles con un paso;
    # no recorta variabilidad clínica moderada.
    good = np.isfinite(intervals) & (intervals >= max(0.18, 0.45 * med)) & (intervals <= min(2.0, 1.80 * med))
    intervals2 = intervals[good]
    transitions = [(sides[i], sides[i+1]) for i in range(len(intervals)) if good[i]]

    if len(intervals2) < 3 or np.nanmean(intervals2) <= 0:
        return {
            "cadence": np.nan, "mean_step": np.nan, "cv": np.nan, "asym": np.nan,
            "events": cleaned, "n_intervals": int(len(intervals2)), "quality": "No fiable",
            "reason": "insuficientes intervalos de paso plausibles"
        }

    mean_step = float(np.nanmean(intervals2))
    cadence = float(60.0 / mean_step)
    cv = float(np.nanstd(intervals2, ddof=1) / mean_step * 100.0) if len(intervals2) >= 3 else np.nan

    lr = [v for v, tr in zip(intervals2, transitions) if tr == ("L", "R")]
    rl = [v for v, tr in zip(intervals2, transitions) if tr == ("R", "L")]
    if len(lr) >= 2 and len(rl) >= 2:
        ml, mr = float(np.mean(lr)), float(np.mean(rl))
        asym = abs(ml - mr) / ((ml + mr) / 2.0) * 100.0 if (ml + mr) > 0 else np.nan
    else:
        asym = np.nan

    quality = "Alta" if len(intervals2) >= 6 else "Moderada"
    return {
        "cadence": cadence, "mean_step": mean_step, "cv": cv, "asym": asym,
        "events": cleaned, "n_intervals": int(len(intervals2)), "quality": quality,
        "reason": f"{len(intervals2)} intervalos derivados de contactos I/D validados"
    }




def _canonical_gait_timeline(L, R, fps):
    """
    v0.9.2 · Línea temporal anatómica canónica.

    Construye una única secuencia temporal de contactos iniciales (IC)
    izquierdos/derechos procedentes de LOS MISMOS ciclos IC→TO→IC usados
    para apoyo/oscilación.

    Principios:
    - exige alternancia anatómica L-R-L-R;
    - no usa pares/impares como sustituto de lateralidad;
    - prioriza intervalos próximos a 1/2 del ciclo ipsilateral mediano;
    - conserva la secuencia temporal de mayor coherencia;
    - no elimina un ciclo únicamente porque aumente el CV.
    """
    fps = float(max(fps, 1.0))
    events = []
    for i in np.asarray(L.get("ic", []), dtype=int):
        events.append((int(i), "L"))
    for i in np.asarray(R.get("ic", []), dtype=int):
        events.append((int(i), "R"))
    events.sort(key=lambda z: (z[0], z[1]))

    raw_cycles = np.r_[
        np.asarray(L.get("cycle", []), dtype=float),
        np.asarray(R.get("cycle", []), dtype=float)
    ]
    raw_cycles = raw_cycles[np.isfinite(raw_cycles) & (raw_cycles >= 0.55) & (raw_cycles <= 2.50)]
    stride_ref = float(np.nanmedian(raw_cycles)) if len(raw_cycles) else np.nan
    step_ref = stride_ref / 2.0 if np.isfinite(stride_ref) and stride_ref > 0 else np.nan

    if len(events) < 4:
        return {
            "events": events, "intervals": np.asarray([], float),
            "cadence": np.nan, "mean_step": np.nan,
            "asym": np.nan, "lr_mean": np.nan, "rl_mean": np.nan,
            "n_intervals": 0, "quality": "No fiable",
            "reason": "menos de 4 IC anatómicos disponibles",
            "stride_ref": stride_ref,
        }

    # Elimina duplicados del mismo lado extremadamente próximos.
    dedup = []
    duplicate_gap = max(1, int(round(0.16 * fps)))
    for ev in events:
        if dedup and ev[1] == dedup[-1][1] and ev[0] - dedup[-1][0] < duplicate_gap:
            continue
        dedup.append(ev)
    events = dedup

    # Intervalo de paso plausible. Es amplio para no borrar variabilidad clínica,
    # pero evita saltos manifiestamente incompatibles con un paso.
    if np.isfinite(step_ref):
        lo_step = max(0.22, 0.58 * step_ref)
        hi_step = min(1.35, 1.55 * step_ref)
    else:
        lo_step, hi_step = 0.22, 1.35

    # Programación dinámica: mejor cadena alternante.
    # score alto = más eventos + menor desviación respecto al paso de referencia.
    n = len(events)
    best_score = np.ones(n, dtype=float)
    prev_idx = np.full(n, -1, dtype=int)
    chain_len = np.ones(n, dtype=int)

    for j in range(n):
        fj, sj = events[j]
        for i in range(j):
            fi, si = events[i]
            if si == sj:
                continue
            dt = (fj - fi) / fps
            if not (lo_step <= dt <= hi_step):
                continue
            penalty = 0.0
            if np.isfinite(step_ref) and step_ref > 0:
                penalty = min(abs(dt - step_ref) / step_ref, 1.0)
            candidate = best_score[i] + 1.0 - 0.18 * penalty
            if candidate > best_score[j]:
                best_score[j] = candidate
                prev_idx[j] = i
                chain_len[j] = chain_len[i] + 1

    end = int(np.argmax(best_score + 0.02 * chain_len))
    idxs = []
    k = end
    while k >= 0:
        idxs.append(k)
        k = int(prev_idx[k])
    idxs = idxs[::-1]
    chain = [events[i] for i in idxs]

    if len(chain) < 4:
        return {
            "events": chain, "intervals": np.asarray([], float),
            "cadence": np.nan, "mean_step": np.nan,
            "asym": np.nan, "lr_mean": np.nan, "rl_mean": np.nan,
            "n_intervals": max(0, len(chain)-1), "quality": "No fiable",
            "reason": "no se obtuvo una cadena L-R alternante suficientemente larga",
            "stride_ref": stride_ref,
        }

    frames = np.asarray([e[0] for e in chain], dtype=float)
    sides = [e[1] for e in chain]
    intervals = np.diff(frames) / fps
    transitions = [(sides[i], sides[i+1]) for i in range(len(sides)-1)]

    good = np.isfinite(intervals) & (intervals >= lo_step) & (intervals <= hi_step)
    intervals_good = intervals[good]
    trans_good = [tr for tr, ok in zip(transitions, good) if ok]

    if len(intervals_good) < 3 or float(np.nanmean(intervals_good)) <= 0:
        return {
            "events": chain, "intervals": intervals_good,
            "cadence": np.nan, "mean_step": np.nan,
            "asym": np.nan, "lr_mean": np.nan, "rl_mean": np.nan,
            "n_intervals": int(len(intervals_good)), "quality": "No fiable",
            "reason": "menos de 3 intervalos de paso anatómicos plausibles",
            "stride_ref": stride_ref,
        }

    mean_step = float(np.nanmean(intervals_good))
    cadence = float(60.0 / mean_step)

    lr = np.asarray([v for v, tr in zip(intervals_good, trans_good) if tr == ("L","R")], float)
    rl = np.asarray([v for v, tr in zip(intervals_good, trans_good) if tr == ("R","L")], float)
    lr_mean = float(np.nanmean(lr)) if len(lr) else np.nan
    rl_mean = float(np.nanmean(rl)) if len(rl) else np.nan

    if len(lr) >= 2 and len(rl) >= 2 and np.isfinite(lr_mean) and np.isfinite(rl_mean):
        denom = (lr_mean + rl_mean) / 2.0
        asym = abs(lr_mean - rl_mean) / denom * 100.0 if denom > 0 else np.nan
    else:
        asym = np.nan

    if len(intervals_good) >= 7 and len(lr) >= 3 and len(rl) >= 3:
        quality = "Alta"
    elif len(intervals_good) >= 5:
        quality = "Moderada"
    else:
        quality = "Baja"

    return {
        "events": chain,
        "intervals": intervals_good,
        "cadence": cadence,
        "mean_step": mean_step,
        "asym": asym,
        "lr_mean": lr_mean,
        "rl_mean": rl_mean,
        "n_lr": int(len(lr)),
        "n_rl": int(len(rl)),
        "n_intervals": int(len(intervals_good)),
        "quality": quality,
        "reason": f"{len(intervals_good)} pasos en cadena anatómica alternante",
        "stride_ref": stride_ref,
    }


def _raw_stride_cadence(L, R):
    """
    Cadencia de control desde ciclos ipsilaterales SIN usar el filtro del CV.
    Así un ciclo estadísticamente variable no altera silenciosamente el ritmo.
    """
    vals = np.r_[
        np.asarray(L.get("cycle", []), dtype=float),
        np.asarray(R.get("cycle", []), dtype=float)
    ]
    vals = vals[np.isfinite(vals) & (vals >= 0.55) & (vals <= 2.50)]
    if len(vals) < 2:
        return np.nan, np.nan, int(len(vals))
    mean_stride = float(np.nanmean(vals))
    cadence = float(120.0 / mean_stride) if mean_stride > 0 else np.nan
    return cadence, mean_stride, int(len(vals))


def _temporal_closure_from_masks(lmask, rmask, L, R, straight_mask, fps):
    """
    Control físico independiente v0.9.2.

    Para marcha sin fase aérea relevante:
        stance_L + stance_R = ciclo * (1 + doble_apoyo_fracción)

    Permite estimar qué cadencia sería compatible simultáneamente con
    tiempos de apoyo y solapamiento bipodal observados.
    """
    cov_l = np.asarray(L.get("coverage_mask", np.ones(len(lmask), dtype=bool)), bool)
    cov_r = np.asarray(R.get("coverage_mask", np.ones(len(rmask), dtype=bool)), bool)
    valid = cov_l & cov_r & np.asarray(straight_mask, bool)

    if not valid.any():
        return {
            "double_raw": np.nan, "flight_pct": np.nan,
            "stance_pct_l": np.nan, "stance_pct_r": np.nan,
            "double_expected": np.nan, "double_discrepancy": np.nan,
            "closure_stride": np.nan, "closure_cadence": np.nan,
        }

    both = np.asarray(lmask, bool) & np.asarray(rmask, bool)
    none = (~np.asarray(lmask, bool)) & (~np.asarray(rmask, bool))
    double_raw = float(np.mean(both[valid]) * 100.0)
    flight_pct = float(np.mean(none[valid]) * 100.0)

    st_l = float(np.nanmean(L.get("stance", []))) if len(L.get("stance", [])) else np.nan
    st_r = float(np.nanmean(R.get("stance", []))) if len(R.get("stance", [])) else np.nan
    sp_l = float(np.nanmean(L.get("stance_pct", []))) if len(L.get("stance_pct", [])) else np.nan
    sp_r = float(np.nanmean(R.get("stance_pct", []))) if len(R.get("stance_pct", [])) else np.nan

    double_expected = max(0.0, sp_l + sp_r - 100.0) if np.isfinite(sp_l) and np.isfinite(sp_r) else np.nan
    double_discrepancy = abs(double_raw - double_expected) if np.isfinite(double_raw) and np.isfinite(double_expected) else np.nan

    closure_stride = np.nan
    closure_cadence = np.nan
    if (
        np.isfinite(st_l) and np.isfinite(st_r) and
        np.isfinite(double_raw) and
        np.isfinite(flight_pct) and flight_pct <= 5.0
    ):
        denom = 1.0 + double_raw / 100.0
        if denom > 0:
            closure_stride = float((st_l + st_r) / denom)
            if closure_stride > 0:
                closure_cadence = float(120.0 / closure_stride)

    return {
        "double_raw": double_raw,
        "flight_pct": flight_pct,
        "stance_pct_l": sp_l,
        "stance_pct_r": sp_r,
        "double_expected": double_expected,
        "double_discrepancy": double_discrepancy,
        "closure_stride": closure_stride,
        "closure_cadence": closure_cadence,
    }


def _pct_disagreement(a, b):
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return np.nan
    return float(abs(a-b) / ((a+b)/2.0) * 100.0)


def _cycle_timing_metrics(L, R, fps):
    """
    v0.9.2 · Estimador robusto de VARIABILIDAD por lado.

    Objetivos:
    - Estimar el CV sin mezclar directamente ambas piernas.
    - Evitar inflar el CV mezclando directamente ciclos izquierdos y derechos.
    - Rechazar ciclos atípicos de forma robusta por lado.
    - Informar tamaño muestral y confianza.
    """

    def _robust_side(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        if len(x) == 0:
            return {
                "raw": x, "clean": x, "mean": np.nan, "sd": np.nan, "cv": np.nan,
                "n_raw": 0, "n_clean": 0, "outliers": 0
            }

        # Filtro fisiológicamente amplio inicial.
        x = x[(x >= 0.55) & (x <= 2.50)]
        if len(x) == 0:
            return {
                "raw": x, "clean": x, "mean": np.nan, "sd": np.nan, "cv": np.nan,
                "n_raw": 0, "n_clean": 0, "outliers": 0
            }

        raw = x.copy()
        med = float(np.nanmedian(x))
        mad = float(np.nanmedian(np.abs(x - med)))

        if len(x) >= 4 and mad > 1e-9:
            robust_z = 0.6745 * np.abs(x - med) / mad
            keep = robust_z <= 3.5
            x = x[keep]

        # Segundo filtro relativo conservador para evitar falsos IC→IC.
        if len(x) >= 3:
            med2 = float(np.nanmedian(x))
            lo = max(0.55, 0.75 * med2)
            hi = min(2.50, 1.25 * med2)
            x = x[(x >= lo) & (x <= hi)]

        mean = float(np.nanmean(x)) if len(x) else np.nan
        sd = float(np.nanstd(x, ddof=1)) if len(x) >= 2 else np.nan
        cv = float(sd / mean * 100.0) if np.isfinite(sd) and np.isfinite(mean) and mean > 0 else np.nan

        return {
            "raw": raw,
            "clean": x,
            "mean": mean,
            "sd": sd,
            "cv": cv,
            "n_raw": int(len(raw)),
            "n_clean": int(len(x)),
            "outliers": int(max(0, len(raw) - len(x))),
        }

    left = _robust_side(L.get("cycle", []))
    right = _robust_side(R.get("cycle", []))

    # Cadencia global desde la duración media de zancada por lado.
    side_means = [v for v in (left["mean"], right["mean"]) if np.isfinite(v)]
    if len(side_means) == 0:
        return {
            "cadence": np.nan, "mean_stride": np.nan, "cv": np.nan,
            "cv_left": np.nan, "cv_right": np.nan,
            "l_mean_stride": np.nan, "r_mean_stride": np.nan,
            "n_left": left["n_clean"], "n_right": right["n_clean"],
            "n_cycles": left["n_clean"] + right["n_clean"],
            "outliers_left": left["outliers"], "outliers_right": right["outliers"],
            "quality": "No fiable",
            "reason": "sin ciclos IC→IC válidos"
        }

    # Media ponderada por número de ciclos válidos.
    num = 0.0
    den = 0
    for side in (left, right):
        if np.isfinite(side["mean"]) and side["n_clean"] > 0:
            num += side["mean"] * side["n_clean"]
            den += side["n_clean"]
    mean_stride = float(num / den) if den > 0 else float(np.nanmean(side_means))
    cadence = float(120.0 / mean_stride) if mean_stride > 0 else np.nan

    # CV global robusto = promedio ponderado de CV izquierdo y derecho.
    cv_num = 0.0
    cv_den = 0
    for side in (left, right):
        if np.isfinite(side["cv"]) and side["n_clean"] >= 2:
            cv_num += side["cv"] * side["n_clean"]
            cv_den += side["n_clean"]
    cv_global = float(cv_num / cv_den) if cv_den > 0 else np.nan

    nL, nR = left["n_clean"], right["n_clean"]
    n_total = nL + nR

    # Confianza explícita según tamaño muestral bilateral.
    if nL >= 4 and nR >= 4:
        quality = "Alta"
    elif nL >= 3 and nR >= 3:
        quality = "Moderada"
    elif nL >= 2 and nR >= 2:
        quality = "Baja"
    else:
        quality = "Muy baja"

    return {
        "cadence": cadence,
        "mean_stride": mean_stride,
        "cv": cv_global,
        "cv_left": left["cv"],
        "cv_right": right["cv"],
        "l_mean_stride": left["mean"],
        "r_mean_stride": right["mean"],
        "n_left": nL,
        "n_right": nR,
        "n_cycles": n_total,
        "outliers_left": left["outliers"],
        "outliers_right": right["outliers"],
        "quality": quality,
        "reason": (
            f"{nL} ciclos izquierdos + {nR} derechos tras filtrado robusto; "
            f"outliers excluidos L={left['outliers']}, R={right['outliers']}"
        )
    }


def _run_durations(mask, fps, min_s=0.15, max_s=10.0):
    vals = []
    for a, b in _bool_runs(mask):
        d = (b - a) / float(fps)
        if min_s <= d <= max_s:
            vals.append(d)
    return np.asarray(vals, float)


def _medial_knee_deviation_angle(row, side):
    hx,hy=row[f"{side}Hip_x"],row[f"{side}Hip_y"]
    kx,ky=row[f"{side}Knee_x"],row[f"{side}Knee_y"]
    ax,ay=row[f"{side}Ankle_x"],row[f"{side}Ankle_y"]
    midx=(row.LHip_x+row.RHip_x)/2.0
    if not all(np.isfinite([hx,hy,kx,ky,ax,ay,midx])) or abs(ay-hy)<1e-6:
        return np.nan
    t=(ky-hy)/(ay-hy)
    line_x=hx+t*(ax-hx)
    toward_mid=(kx-line_x)*(midx-line_x)>0
    if not toward_mid:
        return 0.0
    ka=point_angle(hx,hy,kx,ky,ax,ay)
    return abs(180.0-ka) if np.isfinite(ka) else np.nan

def add_frontal_advanced(seg, fps, scale_cm_per_px=0.0):
    seg=seg.copy()
    hipx=(seg.LHip_x.to_numpy(float)+seg.RHip_x.to_numpy(float))/2.0
    shx=(seg.LShoulder_x.to_numpy(float)+seg.RShoulder_x.to_numpy(float))/2.0
    # Proxy del centro de masa: combinación tronco-pelvis, no CoM segmentario 3D.
    seg["com_proxy_x"] = 0.65*hipx + 0.35*shx
    lmask, _, _ = _support_mask_2d(seg, "L", fps); rmask, _, _ = _support_mask_2d(seg, "R", fps)
    seg["L_support_2d"]=lmask; seg["R_support_2d"]=rmask
    lx,_=_foot_centroid(seg,"L"); rx,_=_foot_centroid(seg,"R")
    seg["bos_px"]=np.abs(rx-lx)
    seg["L_dynamic_valgus"]=[_medial_knee_deviation_angle(r,"L") for _,r in seg.iterrows()]
    seg["R_dynamic_valgus"]=[_medial_knee_deviation_angle(r,"R") for _,r in seg.iterrows()]
    if scale_cm_per_px and scale_cm_per_px>0:
        seg["com_proxy_cm"]=(seg.com_proxy_x-np.nanmedian(seg.com_proxy_x))*float(scale_cm_per_px)
        seg["bos_cm"]=seg.bos_px*float(scale_cm_per_px)
    else:
        seg["com_proxy_cm"]=np.nan
        seg["bos_cm"]=np.nan
    return seg

def visibility_pct(seg, names, threshold=0.5):
    cols = [f"{n}_score" for n in names if f"{n}_score" in seg.columns]
    if not cols:
        return np.nan
    return float((seg[cols].min(axis=1) >= threshold).mean() * 100)


def _sync_signal(df):
    """Señal corporal vertical normalizada para estimar desfase temporal entre cámaras."""
    if df is None or df.empty:
        return np.array([]), np.array([])
    hip_y = (df["LHip_y"].to_numpy(float) + df["RHip_y"].to_numpy(float)) / 2.0
    sh_y = (df["LShoulder_y"].to_numpy(float) + df["RShoulder_y"].to_numpy(float)) / 2.0
    torso = np.abs(hip_y - sh_y)
    torso[torso < 1.0] = np.nan
    sig = hip_y / torso
    sig = pd.Series(sig).interpolate(limit_direction="both").rolling(7, center=True, min_periods=1).mean().to_numpy()
    sig = np.gradient(sig)
    return df["frame"].to_numpy(float), sig


def estimate_sync_offset(df1, fps1, df2, fps2, max_offset_s=2.0, resample_hz=50):
    """
    Estima desfase cam02 vs cam01 por correlación de movimiento vertical corporal.
    offset_s > 0: el mismo evento aparece más tarde en cam02; para alinear, avanzar/recortar cam02 ese tiempo.
    Es una heurística experimental, no sustituto de sincronización hardware.
    """
    f1, s1 = _sync_signal(df1); f2, s2 = _sync_signal(df2)
    if len(s1) < 20 or len(s2) < 20 or fps1 <= 0 or fps2 <= 0:
        return 0.0, np.nan, "No calculable"
    t1 = f1 / float(fps1); t2 = f2 / float(fps2)
    dur = min(float(np.nanmax(t1)), float(np.nanmax(t2)))
    if dur < 2.0:
        return 0.0, np.nan, "No calculable"
    grid = np.arange(0.0, dur, 1.0 / resample_hz)
    a = np.interp(grid, t1, s1); b = np.interp(grid, t2, s2)
    a = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-8)
    b = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-8)
    max_k = int(round(max_offset_s * resample_hz))
    best_k, best_corr = 0, -np.inf
    for k in range(-max_k, max_k + 1):
        if k > 0:
            x, y = a[:-k], b[k:]
        elif k < 0:
            x, y = a[-k:], b[:k]
        else:
            x, y = a, b
        if len(x) < resample_hz:
            continue
        c = np.corrcoef(x, y)[0,1]
        if np.isfinite(c) and c > best_corr:
            best_corr, best_k = float(c), int(k)
    offset_s = best_k / float(resample_hz)
    quality = "Alta" if best_corr >= 0.65 else ("Moderada" if best_corr >= 0.40 else "Baja")
    return float(offset_s), float(best_corr), quality


def prefix_metrics(metrics, prefix, label_prefix):
    out = []
    for m in metrics:
        x = dict(m)
        x["key"] = f"{prefix}_{m['key']}"
        x["label"] = f"{label_prefix} · {m['label']}"
        out.append(x)
    return out


def camera_pose_quality(df):
    if df is None or df.empty:
        return {"tracking": np.nan, "good_frames": np.nan, "foot_visibility": np.nan, "lower_visibility": np.nan}
    score_cols = [f"{n}_score" for n in LOWER_BODY if f"{n}_score" in df.columns]
    tracking = float(df[score_cols].mean(axis=1).mean()) if score_cols else np.nan
    good = float((df[score_cols].min(axis=1) >= 0.5).mean() * 100) if score_cols else np.nan
    foot = visibility_pct(df, FOOT_POINTS, 0.5)
    lower = visibility_pct(df, LOWER_BODY, 0.5)
    return {"tracking": tracking, "good_frames": good, "foot_visibility": foot, "lower_visibility": lower}


def readiness_3d(mode, q1, q2, sync_quality, calibration_name):
    reasons = []
    if not mode.startswith("2 cámaras"):
        reasons.append("Se requieren dos cámaras.")
    if not calibration_name:
        reasons.append("Falta un perfil de calibración 2 cámaras.")
    if sync_quality not in ("Alta", "Moderada"):
        reasons.append("La sincronización automática es insuficiente o no calculable.")
    for idx, q in enumerate((q1, q2), start=1):
        lv = q.get("lower_visibility", np.nan) if q else np.nan
        if not np.isfinite(lv) or lv < 70:
            reasons.append(f"Visibilidad de tren inferior insuficiente en cámara {idx} (<70%).")
    return len(reasons) == 0, reasons



def _edges(mask):
    """Devuelve inicios y finales de periodos True. Índices relativos al segmento."""
    m=np.asarray(mask,dtype=bool)
    if len(m)==0:
        return np.array([],dtype=int), np.array([],dtype=int)
    d=np.diff(m.astype(int), prepend=int(m[0]))
    starts=np.where(d==1)[0].tolist()
    ends=np.where(d==-1)[0].tolist()
    if m[0]: starts=[0]+starts
    if m[-1]: ends=ends+[len(m)]
    return np.asarray(starts,dtype=int), np.asarray(ends,dtype=int)


def _nearest_valid_indices(starts, ends, n):
    out=[]
    for ic in starts:
        offs=ends[ends>ic]
        if len(offs):
            out.append((int(ic), int(min(offs[0], n-1))))
    return out


def _mean_at_indices(arr, idxs):
    a=np.asarray(arr,float)
    vals=[a[int(i)] for i in idxs if 0<=int(i)<len(a) and np.isfinite(a[int(i)])]
    return float(np.nanmean(vals)) if vals else np.nan


def _window_stat(arr, starts, cycles, lo_pct, hi_pct, fn="max"):
    """Estadística en ventana porcentual del ciclo, usando IC consecutivos como ciclo."""
    a=np.asarray(arr,float); vals=[]
    for j in range(len(starts)-1):
        i0,i1=int(starts[j]),int(starts[j+1])
        if i1-i0<5: continue
        lo=i0+int(round((i1-i0)*lo_pct/100.0)); hi=i0+int(round((i1-i0)*hi_pct/100.0))
        lo=max(i0,min(lo,len(a)-1)); hi=max(lo+1,min(hi,len(a)))
        x=a[lo:hi]; x=x[np.isfinite(x)]
        if not len(x): continue
        vals.append(float(np.nanmax(x) if fn=="max" else np.nanmin(x) if fn=="min" else np.nanmean(x)))
    return float(np.nanmean(vals)) if vals else np.nan



def compute_gait_phase_metrics(seg, fps, is_frontal, expected_stride_s=np.nan,
                               support_summary_l=None, support_summary_r=None,
                               temporal_coherence_ok=True):
    """
    v0.9.2: fases del ciclo a partir de estados de apoyo estabilizados.
    IC/TO siguen siendo eventos cinemáticos 2D estimados.
    """
    lmask = np.asarray(seg["L_support_2d"], bool) if "L_support_2d" in seg else _support_mask_2d(seg, "L", fps, expected_stride_s)[0]
    rmask = np.asarray(seg["R_support_2d"], bool) if "R_support_2d" in seg else _support_mask_2d(seg, "R", fps, expected_stride_s)[0]

    L = support_summary_l or _support_cycle_summary(lmask, fps, expected_stride_s)
    R = support_summary_r or _support_cycle_summary(rmask, fps, expected_stride_s)
    reliable = L["quality"] in ("Alta", "Moderada") and R["quality"] in ("Alta", "Moderada")

    if reliable:
        # v0.9.2: el doble apoyo se calcula SOLO en el dominio cubierto
        # simultáneamente por ciclos válidos de ambos lados y por marcha rectilínea.
        cov_l = np.asarray(L.get("coverage_mask", np.ones(len(lmask), dtype=bool)), bool)
        cov_r = np.asarray(R.get("coverage_mask", np.ones(len(rmask), dtype=bool)), bool)
        valid_domain = cov_l & cov_r
        if "straight_walking_valid" in seg:
            valid_domain &= np.asarray(seg["straight_walking_valid"], bool)

        both = lmask & rmask
        none = (~lmask) & (~rmask)

        if valid_domain.any():
            double_pct_raw = float(np.mean(both[valid_domain]) * 100.0)
            double_s_raw = float(np.sum(both[valid_domain]) / float(fps))
            flight_pct = float(np.mean(none[valid_domain]) * 100.0)
        else:
            double_pct_raw = double_s_raw = flight_pct = np.nan

        swing_l = float(np.nanmean(L["swing"])) if len(L["swing"]) else np.nan
        swing_r = float(np.nanmean(R["swing"])) if len(R["swing"]) else np.nan
        swing_asym = abs(swing_l - swing_r) / ((swing_l + swing_r) / 2.0) * 100.0 if np.isfinite(swing_l) and np.isfinite(swing_r) and (swing_l + swing_r) > 0 else np.nan
        stance_pct_l = float(np.nanmean(L["stance_pct"])) if len(L["stance_pct"]) else np.nan
        stance_pct_r = float(np.nanmean(R["stance_pct"])) if len(R["stance_pct"]) else np.nan
        stance_pct = float(np.nanmean([stance_pct_l, stance_pct_r])) if np.isfinite(stance_pct_l) and np.isfinite(stance_pct_r) else np.nan
        swing_pct = 100.0 - stance_pct if np.isfinite(stance_pct) else np.nan

        # Control físico interno: en marcha ordinaria el detector no debería
        # producir una fase aérea apreciable. Si ocurre, no se publica doble apoyo.
        expected_double = max(0.0, stance_pct_l + stance_pct_r - 100.0) if np.isfinite(stance_pct_l) and np.isfinite(stance_pct_r) else np.nan
        ds_discrepancy = abs(double_pct_raw - expected_double) if np.isfinite(double_pct_raw) and np.isfinite(expected_double) else np.nan
        ds_ok = (
            np.isfinite(double_pct_raw)
            and np.isfinite(flight_pct)
            and flight_pct <= 5.0
            and (not np.isfinite(ds_discrepancy) or ds_discrepancy <= 5.0)
            and bool(temporal_coherence_ok)
        )
        if ds_ok:
            double_pct = double_pct_raw
            double_s = double_s_raw
            phase_quality = "Experimental · ciclos rectilíneos validados"
        else:
            double_pct = double_s = np.nan
            phase_quality = "Parcialmente fiable · doble apoyo suprimido por coherencia temporal/física"
    else:
        double_pct = double_s = swing_l = swing_r = swing_asym = stance_pct = swing_pct = np.nan
        flight_pct = expected_double = ds_discrepancy = np.nan
        phase_quality = "No fiable"

    out = [
        {"key":"temporal_segmentation_score","label":"Consistencia de segmentación apoyo/oscilación","value":float(min(L["score"], R["score"])),"unit":"/100","quality":"Alta" if reliable and min(L["score"],R["score"])>=75 else ("Moderada" if reliable else "No fiable"),"notes":f"Control ciclo a ciclo. Izq: {L['reason']}. Der: {R['reason']}. No es un índice clínico."},
        {"key":"stance_pct_2d","label":"Fase de apoyo estimada","value":stance_pct,"unit":"% ciclo","quality":phase_quality,"notes":"IC→TO estimados mediante estados temporales estabilizados; referencia ~60% solo como contexto."},
        {"key":"swing_pct_2d","label":"Fase de oscilación estimada","value":swing_pct,"unit":"% ciclo","quality":phase_quality,"notes":"TO→siguiente IC; se anula si falla la consistencia temporal."},
        {"key":"double_support_pct_2d","label":"Doble apoyo estimado","value":double_pct,"unit":"% dominio válido","quality":phase_quality,"notes":"v0.9.2: intersección directa de apoyos dentro del dominio rectilíneo común. Se suprime si hay fase aérea >5%, discordancia >5 puntos porcentuales respecto a la ocupación de apoyo o fallo de coherencia temporal global."},
        {"key":"double_support_expected_from_stance_pct","label":"Doble apoyo esperado por ocupación de apoyo","value":expected_double,"unit":"% ciclo","quality":"Control interno","notes":"Máx(0, apoyo_I% + apoyo_D% − 100). Se usa como comprobación matemática, no como medida clínica independiente."},
        {"key":"unsupported_flight_pct_2d","label":"Frames sin apoyo detectado","value":flight_pct,"unit":"% dominio válido","quality":"Control interno","notes":"En marcha ordinaria un valor relevante sugiere fallo de segmentación; >5% invalida el doble apoyo."},
        {"key":"double_support_internal_discrepancy_pct","label":"Discrepancia interna de doble apoyo","value":ds_discrepancy,"unit":"puntos %","quality":"Control interno","notes":"Diferencia entre solapamiento observado y esperado por ocupación de apoyo; >10 puntos invalida el doble apoyo."},
        {"key":"double_support_time_2d","label":"Tiempo acumulado de doble apoyo","value":double_s,"unit":"s","quality":phase_quality,"notes":"Tiempo acumulado del segmento; no duración media por ciclo."},
        {"key":"swing_time_l_2d","label":"Tiempo de oscilación izquierdo estimado","value":swing_l,"unit":"s","quality":phase_quality,"notes":"TO izquierdo→siguiente IC izquierdo."},
        {"key":"swing_time_r_2d","label":"Tiempo de oscilación derecho estimado","value":swing_r,"unit":"s","quality":phase_quality,"notes":"TO derecho→siguiente IC derecho."},
        {"key":"swing_asymmetry_2d","label":"Asimetría de oscilación estimada","value":swing_asym,"unit":"%","quality":phase_quality,"notes":"Diferencia relativa D/I solo si la segmentación es consistente."},
        {"key":"initial_contacts_l_n","label":"Contactos iniciales izquierdos estimados","value":float(len(L["ic"])) if reliable else np.nan,"unit":"eventos","quality":phase_quality,"notes":"Eventos cinemáticos 2D estimados; no heel-strikes validados."},
        {"key":"initial_contacts_r_n","label":"Contactos iniciales derechos estimados","value":float(len(R["ic"])) if reliable else np.nan,"unit":"eventos","quality":phase_quality,"notes":"Eventos cinemáticos 2D estimados; no heel-strikes validados."},
    ]

    if not reliable:
        return out

    if is_frontal:
        for side, label, summary in [("L","izquierda",L),("R","derecha",R)]:
            ics = summary["ic"]
            foot = seg[f"{side}_foot_progress_proj"].to_numpy(float)
            rear = seg[f"{side}_rearfoot_tilt_proj"].to_numpy(float)
            valg = seg[f"{side}_dynamic_valgus"].to_numpy(float)
            out += [
                {"key":f"initial_contact_foot_{side.lower()}_deg","label":f"Orientación del pie {label} en contacto inicial estimado","value":_mean_at_indices(foot,ics),"unit":"°","quality":phase_quality,"notes":"Orientación distal 2D en IC estimado; no confirma talón/antepié."},
                {"key":f"initial_contact_rearfoot_{side.lower()}_deg","label":f"Retropié {label} en contacto inicial estimado","value":_mean_at_indices(rear,ics),"unit":"°","quality":phase_quality,"notes":"Inclinación proyectada del retropié en IC estimado."},
                {"key":f"loading_knee_{side.lower()}_deg","label":f"Desviación medial rodilla {label} en respuesta a la carga","value":_window_stat(valg,ics,None,0,10,"max"),"unit":"°","quality":phase_quality,"notes":"Máximo 2D en ventana 0–10% de ciclos aceptados; el valgo real es multiplanar."},
                {"key":f"terminal_foot_{side.lower()}_deg","label":f"Orientación distal pie {label} en pre-oscilación","value":_window_stat(foot,ics,None,50,60,"mean"),"unit":"°","quality":phase_quality,"notes":"Ventana 50–60% solo en ciclos aceptados."},
            ]
    else:
        for side, label, summary in [("L","izquierda",L),("R","derecha",R)]:
            ics = summary["ic"]
            ankle = seg[f"{side}_ankle_angle"].to_numpy(float)
            knee = seg[f"{side}_knee_flex"].to_numpy(float)
            out += [
                {"key":f"initial_contact_foot_{side.lower()}_deg","label":f"Ángulo tobillo-pie {label} en contacto inicial estimado","value":_mean_at_indices(ankle,ics),"unit":"°","quality":phase_quality,"notes":"Ángulo sagital 2D en IC estimado; no confirma heel-strike mediante fuerza."},
                {"key":f"loading_knee_{side.lower()}_deg","label":f"Flexión rodilla {label} en respuesta a la carga","value":_window_stat(knee,ics,None,0,10,"max"),"unit":"°","quality":phase_quality,"notes":"Máxima flexión proyectada en ventana 0–10% de ciclos aceptados."},
                {"key":f"terminal_foot_{side.lower()}_deg","label":f"Ángulo tobillo-pie {label} en pre-oscilación","value":_window_stat(ankle,ics,None,50,60,"mean"),"unit":"°","quality":phase_quality,"notes":"Ventana 50–60% de ciclos aceptados."},
            ]
    return out


def compute_metrics(df, fps, start_frame, end_frame, view, assistive_device="Sin ayuda", scale_cm_per_px=0.0):
    seg = df[(df.frame >= start_frame) & (df.frame <= end_frame)].copy()
    if len(seg) < max(30, int(fps * 2)):
        raise ValueError("El segmento seleccionado es demasiado corto.")
    seg = add_angle_columns(seg)
    is_frontal = "Frontal" in (view or "")
    if is_frontal:
        seg = add_frontal_columns(seg)
        seg = add_frontal_advanced(seg, fps, scale_cm_per_px)

    score_cols = [f"{n}_score" for n in LOWER_BODY]
    mean_tracking = float(seg[score_cols].mean(axis=1).mean())
    good_frames = float((seg[score_cols].min(axis=1) >= 0.5).mean() * 100)
    foot_visible = visibility_pct(seg, FOOT_POINTS, 0.5)
    upper_visible = visibility_pct(seg, UPPER_BODY, 0.5)
    q = quality_label(mean_tracking)
    assisted = assistive_device != "Sin ayuda"

    # v0.8.1 · dominio rectilíneo + ciclos IC→IC coherentes.
    # El giro y las transiciones no deben contaminar CV, asimetría ni doble apoyo.
    straight_mask, straight_info = _automatic_straight_walking_mask(seg, fps)
    seg["straight_walking_valid"] = straight_mask

    # Proxy distal se conserva solo para una primera estimación amplia del periodo.
    ly = (seg.LAnkle_y.to_numpy() + seg.LHeel_y.to_numpy() + seg.LBigToe_y.to_numpy()) / 3.0
    ry = (seg.RAnkle_y.to_numpy() + seg.RHeel_y.to_numpy() + seg.RBigToe_y.to_numpy()) / 3.0
    diff = rolling_smooth(ry - ly, 7)
    crossings = zero_crossings(diff)
    if len(crossings) > 1:
        kept = [crossings[0]]
        min_gap = max(1, int(round(0.25 * fps)))
        for c in crossings[1:]:
            if c - kept[-1] >= min_gap:
                kept.append(c)
        crossings = np.asarray(kept, dtype=int)

    # Periodo inicial robusto para estabilizar el detector de apoyo.
    intervals0 = np.diff(crossings) / fps if len(crossings) >= 2 else np.asarray([])
    mean_alt0 = float(np.nanmedian(intervals0)) if len(intervals0) >= 3 else np.nan
    expected_stride_s = (2.0 * mean_alt0) if np.isfinite(mean_alt0) and mean_alt0 > 0 else np.nan

    l_support, _, _ = _support_mask_2d(seg, "L", fps, expected_stride_s)
    r_support, _, _ = _support_mask_2d(seg, "R", fps, expected_stride_s)
    seg["L_support_2d"] = l_support
    seg["R_support_2d"] = r_support

    l_cycle0 = _support_cycle_summary(l_support, fps, expected_stride_s)
    r_cycle0 = _support_cycle_summary(r_support, fps, expected_stride_s)

    # Excluir ciclos que invaden giro/transiciones.
    l_cycle = _filter_support_summary_to_mask(l_cycle0, straight_mask, fps, min_fraction=0.90)
    r_cycle = _filter_support_summary_to_mask(r_cycle0, straight_mask, fps, min_fraction=0.90)
    support_reliable = l_cycle["quality"] in ("Alta", "Moderada") and r_cycle["quality"] in ("Alta", "Moderada")

    # v0.9.2 · Separación explícita de funciones:
    # - CV: robusto por lado (v0.9.1, preservado).
    # - Cadencia: secuencia anatómica L-R-L-R canónica.
    # - Cadencia ipsilateral: control secundario SIN filtro de outliers del CV.
    # - Cierre físico: apoyo L + apoyo R + doble apoyo observado.
    cycle_timing = _cycle_timing_metrics(l_cycle, r_cycle, fps) if support_reliable else {
        "cadence": np.nan, "mean_stride": np.nan, "cv": np.nan,
        "cv_left": np.nan, "cv_right": np.nan,
        "l_mean_stride": np.nan, "r_mean_stride": np.nan,
        "n_left": 0, "n_right": 0, "n_cycles": 0,
        "outliers_left": 0, "outliers_right": 0,
        "quality": "No fiable",
        "reason": "segmentación apoyo/oscilación insuficiente"
    }

    canonical = _canonical_gait_timeline(l_cycle, r_cycle, fps) if support_reliable else {
        "cadence": np.nan, "mean_step": np.nan, "asym": np.nan,
        "events": [], "n_intervals": 0, "quality": "No fiable",
        "reason": "segmentación apoyo/oscilación insuficiente"
    }

    cadence_steps = canonical.get("cadence", np.nan)
    cadence_stride, mean_stride_raw, n_stride_raw = _raw_stride_cadence(l_cycle, r_cycle) if support_reliable else (np.nan, np.nan, 0)

    closure = _temporal_closure_from_masks(
        l_support, r_support, l_cycle, r_cycle, straight_mask, fps
    ) if support_reliable else {
        "double_raw": np.nan, "flight_pct": np.nan,
        "stance_pct_l": np.nan, "stance_pct_r": np.nan,
        "double_expected": np.nan, "double_discrepancy": np.nan,
        "closure_stride": np.nan, "closure_cadence": np.nan,
    }
    cadence_closure = closure.get("closure_cadence", np.nan)

    # CV permanece exactamente en la lógica robusta v0.9.1.
    cv_alt = cycle_timing.get("cv", np.nan)

    # Concordancias entre tres lecturas del MISMO fenómeno temporal.
    err_step_stride = _pct_disagreement(cadence_steps, cadence_stride)
    err_step_closure = _pct_disagreement(cadence_steps, cadence_closure)
    err_stride_closure = _pct_disagreement(cadence_stride, cadence_closure)

    available_cadences = [x for x in (cadence_steps, cadence_stride, cadence_closure) if np.isfinite(x)]
    pair_errors = [x for x in (err_step_stride, err_step_closure, err_stride_closure) if np.isfinite(x)]
    max_pair_error = float(max(pair_errors)) if pair_errors else np.nan

    # Regla fuerte v0.9.2:
    # con tres estimadores disponibles, los tres deben concordar <=10%.
    # con dos, ambos deben concordar <=10%.
    # con uno solo, no se publica cadencia primaria.
    if len(available_cadences) >= 3:
        temporal_coherence_ok = bool(max_pair_error <= 10.0)
        temporal_coherence_quality = "Alta" if max_pair_error <= 6.0 else ("Moderada" if temporal_coherence_ok else "No fiable")
    elif len(available_cadences) == 2:
        temporal_coherence_ok = bool(max_pair_error <= 10.0)
        temporal_coherence_quality = "Moderada" if temporal_coherence_ok else "No fiable"
    else:
        temporal_coherence_ok = False
        temporal_coherence_quality = "No fiable"

    # El ritmo principal procede de pasos anatómicos; NO del filtrado usado para CV.
    # Si falla el cierre temporal se suprime para evitar una cifra falsa precisa.
    cadence = float(cadence_steps) if temporal_coherence_ok and np.isfinite(cadence_steps) else np.nan
    mean_alt = canonical.get("mean_step", np.nan) if temporal_coherence_ok else np.nan
    mean_stride_cycle = (2.0 * mean_alt) if np.isfinite(mean_alt) else np.nan

    # La asimetría solo se publica si nace de una cadena L/R anatómica y todo el
    # sistema temporal supera el control de coherencia.
    asym = canonical.get("asym", np.nan) if temporal_coherence_ok else np.nan

    if support_reliable:
        stance_l = float(np.nanmean(l_cycle["stance"])) if len(l_cycle["stance"]) else np.nan
        stance_r = float(np.nanmean(r_cycle["stance"])) if len(r_cycle["stance"]) else np.nan
    else:
        stance_l = stance_r = np.nan

    stance_asym = abs(stance_l-stance_r)/((stance_l+stance_r)/2.0)*100.0 if np.isfinite(stance_l) and np.isfinite(stance_r) and (stance_l+stance_r)>0 else np.nan
    stance_longer = 1.0 if np.isfinite(stance_l) and np.isfinite(stance_r) and stance_l>stance_r else (2.0 if np.isfinite(stance_l) and np.isfinite(stance_r) and stance_r>stance_l else 0.0)

    segment_duration_s = float((seg.frame.max() - seg.frame.min() + 1) / fps)
    valid_duration_s = float(np.sum(straight_mask) / fps)

    detected_steps = int(len(canonical.get("events", [])))
    cadence_count_segment = (60.0 * detected_steps / valid_duration_s) if valid_duration_s > 0 and detected_steps > 0 else np.nan
    expected_steps = (cadence * valid_duration_s / 60.0) if np.isfinite(cadence) else np.nan
    consistency_error = (
        abs(detected_steps - expected_steps) / max(expected_steps, 1.0) * 100.0
        if np.isfinite(expected_steps) else np.nan
    )

    if not temporal_coherence_ok:
        consistency_quality = "Revisar"
    elif detected_steps < 4 or not np.isfinite(consistency_error):
        consistency_quality = "Limitada"
    elif consistency_error <= 10:
        consistency_quality = "Alta"
    elif consistency_error <= 20:
        consistency_quality = "Moderada"
    else:
        consistency_quality = "Revisar"

    aid_note = f" Marcha con ayuda técnica: {assistive_device}; revisar oclusiones." if assisted else ""
    metrics = [
        {"key":"tracking_mean","label":"Confianza media del tracking","value":mean_tracking,"unit":"","quality":q,"notes":"Media HALPE26 del tren inferior."+aid_note},
        {"key":"good_frames_pct","label":"Frames con tren inferior visible ≥0,50","value":good_frames,"unit":"%","quality":q,"notes":"Puntos principales del tren inferior ≥0,50."+aid_note},
        {"key":"foot_visibility_pct","label":"Visibilidad de pie/tobillo","value":foot_visible,"unit":"%","quality":quality_label((foot_visible or 0)/100),"notes":"Tobillo, talón y antepié. Fundamental para métricas del pie."+aid_note},
        {"key":"upper_visibility_pct","label":"Visibilidad del tren superior","value":upper_visible,"unit":"%","quality":quality_label((upper_visible or 0)/100),"notes":"Hombros, codos y muñecas; puede disminuir con muletas/caminador."+aid_note},
        {"key":"straight_walking_usable_pct","label":"Tramo rectilíneo utilizable","value":float(straight_info["usable_pct"]),"unit":"% segmento","quality":"Control interno","notes":straight_info["reason"]+"; v0.9.2 excluye giro/transiciones de las métricas temporales sensibles."},
        {"key":"turn_transition_excluded_pct","label":"Giro/transiciones excluidos","value":float(straight_info["excluded_pct"]),"unit":"% segmento","quality":"Control interno","notes":"Exclusión automática por orientación corporal/inversión de trayectoria; no es una métrica clínica."},
        {"key":"cadence_exp","label":"Cadencia estimada","value":cadence,"unit":"pasos/min","quality":temporal_coherence_quality if np.isfinite(cadence) else "No fiable · discordancia temporal","notes":"v0.9.2: 60 / intervalo medio de una cadena anatómica IC izquierda-derecha alternante. Solo se publica si concuerda ≤10% con la cadencia ipsilateral y con el cierre apoyo+doble apoyo."},
        {"key":"step_events_detected","label":"Contactos IC anatómicos en cadena válida","value":float(detected_steps),"unit":"eventos","quality":canonical.get("quality","No fiable"),"notes":"Contactos iniciales L/R derivados de ciclos de apoyo y ordenados en una única secuencia alternante; no son heel-strikes de plataforma de fuerzas."},
        {"key":"segment_duration_s","label":"Duración del segmento analizado","value":segment_duration_s,"unit":"s","quality":"Directa","notes":"Duración temporal total del intervalo seleccionado."},
        {"key":"valid_straight_duration_s","label":"Duración rectilínea utilizable","value":valid_duration_s,"unit":"s","quality":"Control interno","notes":"Tiempo realmente usado tras excluir bordes/giro/transiciones."},
        {"key":"stride_cycle_duration_s","label":"Duración de zancada desde cadena alternante","value":mean_stride_cycle,"unit":"s","quality":temporal_coherence_quality if np.isfinite(mean_stride_cycle) else "No fiable","notes":"Dos intervalos de paso de la línea temporal anatómica v0.9.2; se suprime si falla la coherencia temporal."},
        {"key":"cadence_count_segment","label":"Cadencia por recuento/duración rectilínea","value":cadence_count_segment,"unit":"pasos/min","quality":"Control interno" if np.isfinite(cadence_count_segment) else "No calculable","notes":"60 × contactos aceptados / duración rectilínea utilizable; solo control secundario."},
        {"key":"expected_steps_from_cadence","label":"Eventos esperados por cadencia × duración válida","value":expected_steps,"unit":"eventos","quality":"Control interno" if np.isfinite(expected_steps) else "No calculable","notes":"Comprobación interna sobre el dominio rectilíneo: cadencia × duración válida / 60."},
        {"key":"cadence_contact_crosscheck_error_pct","label":"Discrepancia cadencia pasos vs ciclos ipsilaterales","value":err_step_stride,"unit":"%","quality":"Alta" if np.isfinite(err_step_stride) and err_step_stride<=10 else ("Revisar" if np.isfinite(err_step_stride) else "No calculable"),"notes":"Control v0.9.2 entre la cadena L-R anatómica y los ciclos IC→IC ipsilaterales sin filtrado del CV."},
        {"key":"cadence_closure_crosscheck_error_pct","label":"Discrepancia cadencia pasos vs cierre apoyo/doble apoyo","value":err_step_closure,"unit":"%","quality":"Alta" if np.isfinite(err_step_closure) and err_step_closure<=10 else ("Revisar" if np.isfinite(err_step_closure) else "No calculable"),"notes":"Contrasta la cadencia de pasos con la cadencia físicamente compatible con apoyo izquierdo + apoyo derecho + doble apoyo observado."},
        {"key":"cadence_stride_closure_error_pct","label":"Discrepancia ciclos vs cierre físico","value":err_stride_closure,"unit":"%","quality":"Alta" if np.isfinite(err_stride_closure) and err_stride_closure<=10 else ("Revisar" if np.isfinite(err_stride_closure) else "No calculable"),"notes":"Tercer control independiente de coherencia temporal."},
        {"key":"temporal_coherence_max_error_pct","label":"Discordancia temporal máxima","value":max_pair_error,"unit":"%","quality":temporal_coherence_quality,"notes":"Máxima discrepancia entre cadencia por pasos anatómicos, ciclos ipsilaterales y cierre apoyo/doble apoyo. Umbral operativo de publicación: ≤10%."},
        {"key":"temporal_coherence_flag","label":"Coherencia temporal global","value":1.0 if temporal_coherence_ok else 0.0,"unit":"bool","quality":temporal_coherence_quality,"notes":"1 = cadencia/asimetría/doble apoyo superan el control de coherencia v0.9.2; 0 = se suprimen las métricas incompatibles."},
        {"key":"cadence_candidate_steps","label":"Cadencia candidata por pasos anatómicos","value":cadence_steps,"unit":"pasos/min","quality":"Control interno","notes":"No se interpreta aisladamente; candidato primario antes del cierre de coherencia."},
        {"key":"cadence_candidate_stride","label":"Cadencia candidata por ciclos ipsilaterales","value":cadence_stride,"unit":"pasos/min","quality":"Control interno","notes":"120 / duración media de ciclos IC→IC plausibles, sin aplicar el filtro estadístico usado para CV."},
        {"key":"cadence_candidate_closure","label":"Cadencia candidata por cierre apoyo/doble apoyo","value":cadence_closure,"unit":"pasos/min","quality":"Control interno","notes":"120 / [(apoyo_L + apoyo_R)/(1 + doble_apoyo_fracción)], solo si no existe fase aérea relevante."},
        {"key":"step_count_consistency_error_pct","label":"Discrepancia recuento-cadencia-duración","value":consistency_error,"unit":"%","quality":consistency_quality,"notes":"Diferencia relativa entre eventos detectados y eventos esperados por cadencia × duración. Sirve como control de consistencia, no como validación clínica."},
        {"key":"alternation_interval","label":"Intervalo medio de paso anatómico","value":mean_alt,"unit":"s","quality":temporal_coherence_quality if np.isfinite(mean_alt) else "No calculable","notes":"IC-L→IC-R o IC-R→IC-L de la cadena anatómica v0.9.2."},
        {"key":"regularity_cv","label":"Variabilidad temporal global robusta (CV)","value":cv_alt,"unit":"%","quality":cycle_timing.get("quality","No fiable") if np.isfinite(cv_alt) else "No calculable","notes":"v0.9.1: promedio ponderado de CV izquierdo y derecho calculados por separado sobre ciclos IC→IC filtrados robustamente. No se mezclan directamente ambas piernas."},
        {"key":"regularity_cv_left","label":"CV temporal izquierdo","value":cycle_timing.get("cv_left"),"unit":"%","quality":cycle_timing.get("quality","No fiable"),"notes":"CV de duración IC→IC de la extremidad izquierda tras rechazo robusto de ciclos atípicos."},
        {"key":"regularity_cv_right","label":"CV temporal derecho","value":cycle_timing.get("cv_right"),"unit":"%","quality":cycle_timing.get("quality","No fiable"),"notes":"CV de duración IC→IC de la extremidad derecha tras rechazo robusto de ciclos atípicos."},
        {"key":"valid_cycles_left","label":"Ciclos válidos izquierdos","value":cycle_timing.get("n_left"),"unit":"ciclos","quality":"Control interno","notes":"Número de ciclos IC→IC izquierdos usados para cadencia/CV."},
        {"key":"valid_cycles_right","label":"Ciclos válidos derechos","value":cycle_timing.get("n_right"),"unit":"ciclos","quality":"Control interno","notes":"Número de ciclos IC→IC derechos usados para cadencia/CV."},
        {"key":"cycle_outliers_left","label":"Ciclos atípicos excluidos izquierdos","value":cycle_timing.get("outliers_left"),"unit":"ciclos","quality":"Control interno","notes":"Ciclos descartados por filtro robusto MAD/mediana."},
        {"key":"cycle_outliers_right","label":"Ciclos atípicos excluidos derechos","value":cycle_timing.get("outliers_right"),"unit":"ciclos","quality":"Control interno","notes":"Ciclos descartados por filtro robusto MAD/mediana."},
        {"key":"temporal_asymmetry_exp","label":"Asimetría temporal global","value":asym,"unit":"%","quality":temporal_coherence_quality if np.isfinite(asym) else "No calculable · control temporal no superado","notes":"v0.9.2: compara tiempos IC-L→IC-R frente a IC-R→IC-L de la MISMA cadena anatómica; solo se publica cuando toda la coherencia temporal supera el umbral ≤10%."},
        {"key":"support_segmentation_score_l","label":"Consistencia segmentación apoyo izquierdo","value":float(l_cycle["score"]),"unit":"/100","quality":l_cycle["quality"],"notes":l_cycle["reason"]+"; control matemático, no índice clínico."},
        {"key":"support_segmentation_score_r","label":"Consistencia segmentación apoyo derecho","value":float(r_cycle["score"]),"unit":"/100","quality":r_cycle["quality"],"notes":r_cycle["reason"]+"; control matemático, no índice clínico."},
        {"key":"stance_time_l_2d","label":"Tiempo de apoyo izquierdo estimado 2D","value":stance_l,"unit":"s","quality":"Experimental · ciclo continuo" if np.isfinite(stance_l) else "No fiable","notes":"IC→TO dentro de ciclos continuos estabilizados; se anula si la segmentación no supera el control temporal."},
        {"key":"stance_time_r_2d","label":"Tiempo de apoyo derecho estimado 2D","value":stance_r,"unit":"s","quality":"Experimental · ciclo continuo" if np.isfinite(stance_r) else "No fiable","notes":"IC→TO dentro de ciclos continuos estabilizados; se anula si la segmentación no supera el control temporal."},
        {"key":"stance_asymmetry_2d","label":"Asimetría del tiempo de apoyo estimado 2D","value":stance_asym,"unit":"%","quality":"Experimental · ciclo continuo" if np.isfinite(stance_asym) else "No fiable","notes":"Solo se calcula si ambos lados superan el control de consistencia de ciclo."},
        {"key":"stance_longer_side_code","label":"Extremidad con mayor apoyo estimado (código)","value":stance_longer,"unit":"1=I,2=D","quality":"Experimental" if stance_longer else "No fiable","notes":"1=izquierda; 2=derecha. No equivale a medición con plataforma/plantilla instrumentada."},
    ]
    metrics += compute_gait_phase_metrics(
        seg, fps, is_frontal,
        expected_stride_s=expected_stride_s,
        support_summary_l=l_cycle,
        support_summary_r=r_cycle,
        temporal_coherence_ok=temporal_coherence_ok,
    )

    if not is_frontal:
        for key_base, label_base, col_base in [
            ("knee_flex", "Flexión rodilla", "knee_flex"),
            ("hip_flex", "Flexión cadera", "hip_flex"),
            ("ankle_angle", "Ángulo tobillo-pie", "ankle_angle"),
            ("shoulder_elev", "Elevación hombro", "shoulder_elev"),
        ]:
            vals = {}
            for side, side_name in [("L","izquierda"),("R","derecha")]:
                arr = seg[f"{side}_{col_base}"].to_numpy(dtype=float)
                p95 = float(np.nanpercentile(arr, 95)); rom = robust_rom(arr); vals[side] = p95
                quality = "Condicionada por ayuda técnica" if assisted and key_base=="shoulder_elev" else q
                metrics += [
                    {"key":f"{key_base}_{side.lower()}_p95","label":f"{label_base} {side_name} 2D (P95)","value":p95,"unit":"°","quality":quality,"notes":f"Ángulo 2D proyectado en vista {view}."+aid_note},
                    {"key":f"{key_base}_{side.lower()}_rom","label":f"ROM {label_base.lower()} {side_name} 2D","value":rom,"unit":"°","quality":quality,"notes":"ROM robusto P95-P5; 2D proyectado."+aid_note},
                ]
            metrics.append({"key":f"{key_base}_diff_p95","label":f"Diferencia D/I {label_base.lower()} 2D","value":abs(vals["L"]-vals["R"]),"unit":"°","quality":q,"notes":"Diferencia absoluta P95 D/I; 2D proyectado."})
    else:
        vals = {}
        for side, side_name in [("L","izquierda"),("R","derecha")]:
            kd = seg[f"{side}_frontal_knee_dev"].to_numpy(float)
            fp = seg[f"{side}_foot_progress_proj"].to_numpy(float)
            rf = seg[f"{side}_rearfoot_tilt_proj"].to_numpy(float)
            vals[side] = {
                "knee": float(np.nanpercentile(kd,95)),
                "foot": float(np.nanmedian(fp)),
                "rear": float(np.nanpercentile(np.abs(rf),95)),
            }
            foot_q = q if foot_visible >= 80 else "Baja/condicionada"
            metrics += [
                {"key":f"frontal_knee_dev_{side.lower()}_p95","label":f"Desviación frontal rodilla {side_name} (P95)","value":vals[side]["knee"],"unit":"°","quality":q,"notes":"Magnitud proyectada del eje cadera-rodilla-tobillo. No diagnostica valgo/varo 3D."},
                {"key":f"foot_progress_{side.lower()}_median","label":f"Orientación del pie {side_name} proyectada (mediana)","value":vals[side]["foot"],"unit":"°","quality":foot_q,"notes":"Proxy distal de orientación/rotación en la imagen. No equivale a rotación axial de cadera."},
                {"key":f"rearfoot_tilt_{side.lower()}_p95","label":f"Inclinación retropié {side_name} proyectada (P95 abs.)","value":vals[side]["rear"],"unit":"°","quality":foot_q,"notes":"Eje tobillo-talón proyectado. Puede sugerir cambios de eversión/inversión, pero no mide pronación 3D."},
            ]
        metrics += [
            {"key":"frontal_knee_dev_diff","label":"Diferencia D/I desviación frontal de rodilla","value":abs(vals['L']['knee']-vals['R']['knee']),"unit":"°","quality":q,"notes":"Comparación 2D proyectada."},
            {"key":"foot_progress_diff","label":"Diferencia D/I orientación del pie proyectada","value":abs(vals['L']['foot']-vals['R']['foot']),"unit":"°","quality":q,"notes":"Proxy distal; no atribuir directamente a rotación de cadera."},
            {"key":"rearfoot_tilt_diff","label":"Diferencia D/I inclinación del retropié proyectada","value":abs(vals['L']['rear']-vals['R']['rear']),"unit":"°","quality":q,"notes":"No equivale a pronación clínica."},
            {"key":"pelvis_obliquity_rom","label":"ROM oblicuidad pélvica proyectada","value":robust_rom(seg.pelvis_obliquity),"unit":"°","quality":q,"notes":"P95-P5 de la línea inter-caderas en el plano de imagen."},
            {"key":"shoulder_obliquity_rom","label":"ROM oblicuidad de hombros proyectada","value":robust_rom(seg.shoulder_obliquity),"unit":"°","quality":"Condicionada por ayuda técnica" if assisted else q,"notes":"P95-P5 de la línea biacromial en el plano de imagen."+aid_note},
            {"key":"trunk_lateral_lean_rom","label":"ROM inclinación lateral del tronco proyectada","value":robust_rom(seg.trunk_lateral_lean),"unit":"°","quality":q,"notes":"P95-P5 del eje centro de pelvis-centro de hombros respecto a la vertical de la imagen."},
            {"key":"shoulder_pelvis_rel_rom","label":"ROM relación hombros-pelvis proyectada","value":robust_rom(seg.shoulder_pelvis_rel),"unit":"°","quality":"Condicionada por ayuda técnica" if assisted else q,"notes":"Variación de la diferencia entre oblicuidad de hombros y pelvis; descriptor 2D del acoplamiento tronco-pelvis."+aid_note},
            {"key":"base_width_relative_median","label":"Anchura de base relativa (tobillos/pelvis)","value":float(np.nanmedian(seg.base_width_relative)),"unit":"ratio","quality":q,"notes":"Anchura proyectada normalizada por anchura pélvica; no es distancia métrica sin calibración."},
        ]
        # Parámetros frontales avanzados, condicionados por tracking y escala espacial.
        com_amp_px = robust_rom(seg.com_proxy_x)
        com_cm = robust_rom(seg.com_proxy_cm) if np.isfinite(seg.com_proxy_cm.to_numpy(float)).any() else np.nan
        if support_reliable:
            ds = seg[seg.L_support_2d & seg.R_support_2d]
            bos_cm = float(np.nanmedian(ds.bos_cm)) if len(ds)>=3 and np.isfinite(ds.bos_cm.to_numpy(float)).any() else np.nan
            l_single = seg[seg.L_support_2d & ~seg.R_support_2d]
            r_single = seg[seg.R_support_2d & ~seg.L_support_2d]
            # Convención de pelvis: + = lado izquierdo elevado / derecho descendido; - = lado izquierdo descendido / derecho elevado.
            trend_r_swing = float(max(0.0, np.nanpercentile(l_single.pelvis_obliquity,95))) if len(l_single)>=3 else np.nan
            trend_l_swing = float(max(0.0, -np.nanpercentile(r_single.pelvis_obliquity,5))) if len(r_single)>=3 else np.nan
        else:
            bos_cm = np.nan
            trend_r_swing = trend_l_swing = np.nan
        valg_l = float(np.nanpercentile(seg.L_dynamic_valgus,95)); valg_r = float(np.nanpercentile(seg.R_dynamic_valgus,95))
        # Acoplamiento tronco-pelvis: correlación y desfase por correlación cruzada.
        a = pd.Series(seg.trunk_lateral_lean).interpolate(limit_direction="both").to_numpy(float)
        b = pd.Series(seg.pelvis_obliquity).interpolate(limit_direction="both").to_numpy(float)
        coupling_r = float(np.corrcoef(a,b)[0,1]) if len(a)>10 and np.nanstd(a)>1e-6 and np.nanstd(b)>1e-6 else np.nan
        phase_deg=np.nan
        if len(a)>20 and np.isfinite(mean_alt) and mean_alt>0:
            aa=(a-np.nanmean(a))/(np.nanstd(a)+1e-8); bb=(b-np.nanmean(b))/(np.nanstd(b)+1e-8)
            maxlag=min(int(round(fps)),len(a)//3); bestlag=0; best=-np.inf
            for lag in range(-maxlag,maxlag+1):
                x,y=(aa[:-lag],bb[lag:]) if lag>0 else ((aa[-lag:],bb[:lag]) if lag<0 else (aa,bb))
                if len(x)>10:
                    c=np.corrcoef(x,y)[0,1]
                    if np.isfinite(c) and abs(c)>best: best=abs(c); bestlag=lag
            stride_period = mean_stride_cycle if np.isfinite(mean_stride_cycle) and mean_stride_cycle > 0 else (2.0 * mean_alt)
            phase_deg = float(bestlag / fps / stride_period * 360.0)
            # v0.9.2: fase circular canónica en [-180, +180).
            phase_deg = float(((phase_deg + 180.0) % 360.0) - 180.0)
        metrics += [
            {"key":"com_lateral_excursion_cm","label":"Oscilación lateral CoM proxy 2D","value":com_cm,"unit":"cm","quality":q if np.isfinite(com_cm) else "Requiere escala","notes":"Proxy tronco-pelvis. Se expresa en cm solo con escala espacial 2D calibrada; no es CoM segmentario 3D."},
            {"key":"com_lateral_excursion_px","label":"Oscilación lateral CoM proxy 2D (imagen)","value":com_amp_px,"unit":"px","quality":q,"notes":"Amplitud robusta P95-P5 en imagen; útil para seguimiento si la cámara permanece idéntica."},
            {"key":"bos_width_cm","label":"Ancho de base de sustentación estimado","value":bos_cm,"unit":"cm","quality":q if np.isfinite(bos_cm) else "Requiere escala/apoyo","notes":"Mediana entre centros de ambos pies durante doble apoyo 2D estimado."},
            {"key":"trendelenburg_drop_l_deg","label":"Caída pélvica dinámica lado izquierdo en suspensión","value":trend_l_swing,"unit":"°","quality":"Experimental" if support_reliable and np.isfinite(trend_l_swing) else "No fiable","notes":"Ángulo proyectado durante apoyo derecho; se anula si la segmentación temporal de apoyo no es consistente. No constituye por sí solo diagnóstico de Trendelenburg."},
            {"key":"trendelenburg_drop_r_deg","label":"Caída pélvica dinámica lado derecho en suspensión","value":trend_r_swing,"unit":"°","quality":"Experimental" if support_reliable and np.isfinite(trend_r_swing) else "No fiable","notes":"Ángulo proyectado durante apoyo izquierdo; se anula si la segmentación temporal de apoyo no es consistente. No constituye por sí solo diagnóstico de Trendelenburg."},
            {"key":"dynamic_knee_valgus_l_deg","label":"Valgo dinámico proyectado rodilla izquierda (P95)","value":valg_l,"unit":"°","quality":q,"notes":"Desviación medial proyectada respecto al eje cadera-tobillo; el valgo real es multiplanar."},
            {"key":"dynamic_knee_valgus_r_deg","label":"Valgo dinámico proyectado rodilla derecha (P95)","value":valg_r,"unit":"°","quality":q,"notes":"Desviación medial proyectada respecto al eje cadera-tobillo; el valgo real es multiplanar."},
            {"key":"trunk_pelvis_coupling_r","label":"Acoplamiento intersegmentario tronco-pelvis","value":coupling_r,"unit":"r","quality":q,"notes":"Correlación entre inclinación lateral del tronco y oblicuidad pélvica; descriptor 2D de estrategia de compensación."},
            {"key":"trunk_pelvis_phase_deg","label":"Desfase tronco-pelvis estimado","value":phase_deg,"unit":"° ciclo","quality":"Experimental" if np.isfinite(phase_deg) else "No calculable","notes":"v0.9.2: desfase de correlación cruzada normalizado al ciclo IC→IC y expresado entre −180° y +180°; útil para seguimiento, no diagnóstico aislado."},
        ]

    chart_data = {
        "frame": seg.frame.to_numpy(), "time_s": seg.frame.to_numpy()/fps,
        "Alternancia D-I": diff,
    }
    if is_frontal:
        chart_data.update({
            "Rodilla frontal izquierda": seg.L_frontal_knee_dev,
            "Rodilla frontal derecha": seg.R_frontal_knee_dev,
            "Orientación pie izquierda": seg.L_foot_progress_proj,
            "Orientación pie derecha": seg.R_foot_progress_proj,
            "Retropié izquierda": seg.L_rearfoot_tilt_proj,
            "Retropié derecha": seg.R_rearfoot_tilt_proj,
            "Oblicuidad pélvica": seg.pelvis_obliquity,
            "Oblicuidad de hombros": seg.shoulder_obliquity,
            "Inclinación lateral del tronco": seg.trunk_lateral_lean,
            "Relación hombros-pelvis": seg.shoulder_pelvis_rel,
            "CoM proxy lateral (px)": seg.com_proxy_x,
            "Valgo dinámico proyectado I": seg.L_dynamic_valgus,
            "Valgo dinámico proyectado D": seg.R_dynamic_valgus,
        })
    else:
        chart_data.update({
            "Rodilla izquierda": seg.L_knee_flex, "Rodilla derecha": seg.R_knee_flex,
            "Cadera izquierda": seg.L_hip_flex, "Cadera derecha": seg.R_hip_flex,
            "Tobillo izquierda": seg.L_ankle_angle, "Tobillo derecha": seg.R_ankle_angle,
            "Hombro izquierda": seg.L_shoulder_elev, "Hombro derecha": seg.R_shoulder_elev,
        })
    return metrics, pd.DataFrame(chart_data), seg

def metric_value(metrics, key):
    for m in metrics:
        if m["key"] == key:
            return m.get("value")
    return None


def metric_quality(metrics, key):
    for m in metrics:
        if m["key"] == key:
            return m.get("quality")
    return None


def fmt(v, n=1):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{n}f}"


def biomech_summary(metrics, view="", prefix=""):
    """Resumen descriptivo 2D; evita convertir proxies en diagnósticos."""
    def mv(key):
        return metric_value(metrics, f"{prefix}{key}")
    parts = []
    n = mv("step_events_detected"); dur = mv("segment_duration_s"); cad = mv("cadence_exp")
    expected = mv("expected_steps_from_cadence"); err = mv("step_count_consistency_error_pct")
    if n is not None and dur is not None:
        txt = f"En el segmento de {fmt(dur,2)} s se detectaron {fmt(n,0)} eventos de paso/alternancia."
        if cad is not None and np.isfinite(cad):
            txt += f" La cadencia estimada fue {fmt(cad,1)} pasos/min"
            if expected is not None and np.isfinite(expected):
                txt += f", equivalente a {fmt(expected,1)} eventos esperados en esa duración"
            if err is not None and np.isfinite(err):
                txt += f" (discrepancia interna {fmt(err,1)} %)."
            else:
                txt += "."
        parts.append(txt)
    if "Frontal" in (view or ""):
        pkL, pkR = mv("frontal_knee_dev_l_p95"), mv("frontal_knee_dev_r_p95")
        pel = mv("pelvis_obliquity_rom"); sho = mv("shoulder_obliquity_rom"); tr = mv("trunk_lateral_lean_rom")
        if pkL is not None and pkR is not None:
            parts.append(f"El eje frontal de rodilla mostró P95 de {fmt(pkL,1)}° a izquierda y {fmt(pkR,1)}° a derecha. Son magnitudes proyectadas 2D, no una medición anatómica 3D de valgo/varo.")
        if pel is not None or sho is not None or tr is not None:
            parts.append(f"En el control axial/frontal, el ROM proyectado fue pelvis {fmt(pel,1)}°, hombros {fmt(sho,1)}° y tronco {fmt(tr,1)}°. Estas medidas describen oscilación y alineación en la imagen.")
        rfL, rfR = mv("rearfoot_tilt_l_p95"), mv("rearfoot_tilt_r_p95")
        if rfL is not None and rfR is not None:
            parts.append(f"La inclinación proyectada del retropié alcanzó P95 absoluto de {fmt(rfL,1)}° a izquierda y {fmt(rfR,1)}° a derecha. Puede informar sobre el patrón de inversión/eversión visible, pero no equivale a pronación 3D.")
    else:
        vals = []
        for k,label in [("hip_flex_diff_p95","cadera"),("knee_flex_diff_p95","rodilla"),("ankle_angle_diff_p95","tobillo"),("shoulder_elev_diff_p95","hombro")]:
            v=mv(k)
            if v is not None and np.isfinite(v): vals.append(f"{label} {fmt(v,1)}°")
        if vals:
            parts.append("Las diferencias D/I de P95 en la vista lateral fueron: " + ", ".join(vals) + ". Son diferencias 2D proyectadas y deben contextualizarse con el ciclo de marcha y la calidad del tracking.")
    return parts


def get_point(row, name, min_score=0.25):
    try:
        if row[f"{name}_score"] < min_score:
            return None
        return int(round(row[f"{name}_x"])), int(round(row[f"{name}_y"]))
    except Exception:
        return None


def render_angle_video(video_path, full_df, out_path, view="Lateral", assistive_device="Sin ayuda"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = out_path.with_name(out_path.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    enriched = add_angle_columns(full_df)
    if "Frontal" in (view or ""):
        enriched = add_frontal_columns(enriched)
    indexed = enriched.set_index("frame")
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_no in indexed.index:
            row = indexed.loc[frame_no]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            for a,b in SKELETON:
                pa, pb = get_point(row,a), get_point(row,b)
                if pa and pb: cv2.line(frame, pa, pb, (40,220,80), 2, cv2.LINE_AA)
            for name in HALPE26:
                p = get_point(row,name)
                if p: cv2.circle(frame,p,3,(0,190,255),-1,cv2.LINE_AA)
            if "Frontal" in (view or ""):
                lines = [
                    f"Rod frontal I/D {row['L_frontal_knee_dev']:.0f}/{row['R_frontal_knee_dev']:.0f} deg",
                    f"Pie orient. I/D {row['L_foot_progress_proj']:.0f}/{row['R_foot_progress_proj']:.0f} deg",
                    f"Retropie I/D {row['L_rearfoot_tilt_proj']:.0f}/{row['R_rearfoot_tilt_proj']:.0f} deg",
                    f"Pelvis {row['pelvis_obliquity']:.0f} deg",
                    f"Hombros {row['shoulder_obliquity']:.0f} deg",
                    f"Tronco {row['trunk_lateral_lean']:.0f} deg",
                    "2D proyectado · no rotacion/pronacion 3D",
                ]
            else:
                lines = [
                    f"Cad I/D {row['L_hip_flex']:.0f}/{row['R_hip_flex']:.0f} deg",
                    f"Rod I/D {row['L_knee_flex']:.0f}/{row['R_knee_flex']:.0f} deg",
                    f"Tob I/D {row['L_ankle_angle']:.0f}/{row['R_ankle_angle']:.0f} deg",
                    f"Hom I/D {row['L_shoulder_elev']:.0f}/{row['R_shoulder_elev']:.0f} deg",
                    "2D proyectado",
                ]
            if assistive_device != "Sin ayuda":
                lines.append(f"Ayuda: {assistive_device}")
            y = 32
            for txt in lines:
                cv2.putText(frame, txt, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
                y += 25
        writer.write(frame)
        frame_no += 1
    cap.release(); writer.release()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = f'"{ffmpeg}" -y -loglevel error -i "{raw}" -c:v libx264 -pix_fmt yuv420p -movflags +faststart -an "{out_path}"'
    rc = os.system(cmd)
    try: raw.unlink(missing_ok=True)
    except Exception: pass
    return out_path if rc == 0 and out_path.exists() else None


def cleanup_temp_session(session_dir):
    try:
        shutil.rmtree(session_dir, ignore_errors=True)
        return True
    except Exception:
        return False

def _metric_sentence(metrics, key, title=None, digits=1):
    m=next((x for x in metrics if x.get("key")==key),None)
    if not m or m.get("value") is None:
        return None
    try:
        v=float(m["value"])
        if not np.isfinite(v): return None
    except Exception:
        return None
    label=title or m.get("label",key); unit=m.get("unit","")
    return f"{label}: {v:.{digits}f} {unit}".strip()+f" ({reference_text_for_metric(key)})"

def _fmt_metric_report(metrics, key, label=None, digits=1):
    v=metric_value(metrics,key)
    if v is None or not np.isfinite(v):
        return None
    m=next((x for x in metrics if x.get("key")==key),{})
    unit=m.get("unit","")
    return f"{label or m.get('label',key)}: {float(v):.{digits}f} {unit}".strip()+f" ({reference_text_for_metric(key)})"


def _phase_metric(metrics, base, two_cam=False):
    key=("lateral_"+base) if two_cam and metric_value(metrics,"lateral_"+base) is not None else base
    return key, metric_value(metrics,key)


def generate_reports(metrics, view, patient_code, record_name, assistive_device):
    """Informe clínico estructurado v0.9.2 con cierre temporal, CV robusto y trazabilidad multipersona."""
    two_cam = any(str(m.get("key", "")).startswith("lateral_") for m in metrics)
    tp = "lateral_" if two_cam else ""
    frontal_prefix = "front_" if two_cam else ""

    def val(key):
        return metric_value(metrics, key)

    def choose(base, prefer_frontal=False):
        candidates = []
        if prefer_frontal:
            candidates.append(frontal_prefix + base)
        candidates += [tp + base, base]
        for k in candidates:
            v = val(k)
            if v is not None and np.isfinite(v):
                return k, float(v)
        return candidates[-1], np.nan

    def ref_for(key):
        # Referencias redactadas para el informe; evita convertir medias/IC en umbrales diagnósticos.
        bare = key.replace("front_", "").replace("lateral_", "")
        if bare == "cadence_exp":
            return "Ref. poblacional contextual: 114.95–118.35 pasos/min (IC95% de la media de cadencia habitual al aire libre en adultos aparentemente sanos; Murtagh et al., Sports Med 2021, PMCID PMC7806575). No es un umbral diagnóstico individual."
        if bare == "regularity_cv":
            return "Sin umbral diagnóstico directamente transferible: en v0.9.2 este CV global es un promedio ponderado de los CV izquierdo y derecho calculados por separado sobre ciclos IC→IC rectilíneos validados y filtrados robustamente; no debe compararse directamente con umbrales publicados para CV de tiempo de paso."
        if bare == "temporal_asymmetry_exp":
            return "Sin umbral universal directamente transferible a este detector 2D. En laboratorio la simetría temporal sana suele aproximarse a 1:1; una cohorte de mujeres jóvenes activas mostró baja asimetría a velocidad preferida (PMCID PMC6335661)."
        if bare in {"stance_time_l_2d", "stance_time_r_2d", "stance_asymmetry_2d"}:
            return "Ref. contextual: el apoyo sano es aproximadamente simétrico entre lados; esta es una estimación cinemática 2D experimental, no una medida de plataforma de fuerzas."
        if bare in {"step_events_detected", "segment_duration_s"}:
            return "Referencia metodológica: control de consistencia interna del propio registro; no existe rango normativo clínico aplicable."
        if bare in {"trendelenburg_drop_l_deg", "trendelenburg_drop_r_deg"}:
            return "Sin umbral diagnóstico 2D universal validado para HALPE26. Descriptor proyectado de caída pélvica durante apoyo monopodal; confirmar clínicamente."
        if bare in {"dynamic_knee_valgus_l_deg", "dynamic_knee_valgus_r_deg"}:
            return "Sin rango normativo validado específicamente para HALPE26 2D. Descriptor de desviación medial proyectada; el valgo dinámico real es multiplanar."
        if bare == "trunk_pelvis_coupling_r":
            return "Sin banda normativa universal para este coeficiente 2D. r≈+1 indica acoplamiento lineal en fase; r≈−1, contrafase; el significado depende de la tarea y del contexto clínico."
        if bare == "trunk_pelvis_phase_deg":
            return "Sin banda normativa universal. En v0.9.2 el desfase se expresa de forma circular entre −180° y +180°; interpretar longitudinalmente y junto con la calidad del tracking."
        return "Sin umbral normativo 2D validado directamente transferible a esta métrica; descriptor proyectado que requiere contextualización clínica."

    # -------- datos principales --------
    kcad, cad = choose("cadence_exp")
    kcv, cv = choose("regularity_cv")
    kasym, asym = choose("temporal_asymmetry_exp")
    kn, n_events = choose("step_events_detected")
    kdur, duration = choose("segment_duration_s")
    ksl, stance_l = choose("stance_time_l_2d")
    ksr, stance_r = choose("stance_time_r_2d")
    ksa, stance_asym = choose("stance_asymmetry_2d")
    _, support_score_l = choose("support_segmentation_score_l")
    _, support_score_r = choose("support_segmentation_score_r")
    support_ok = np.isfinite(stance_l) and np.isfinite(stance_r)

    lines = []
    lines.append("INFORME DE ANÁLISIS BIOMECÁNICO DE LA MARCHA (2D)")
    lines.append("")
    lines.append("Ficha del registro:")
    lines.append(f"• Paciente / Código: {patient_code}")
    lines.append(f"• Prueba: Análisis de marcha en vista {view}")
    lines.append(f"• Ayuda técnica: {assistive_device}")
    if np.isfinite(n_events) and np.isfinite(duration):
        lines.append(f"• Consistencia interna: {n_events:.0f} eventos detectados en {duration:.2f} s de registro analizado ({ref_for(kn)})")
    else:
        lines.append("• Consistencia interna: no calculable con los datos disponibles (Referencia metodológica: control interno del registro; no existe rango normativo clínico aplicable).")
    if np.isfinite(support_score_l) and np.isfinite(support_score_r):
        qtmp = "suficiente para estimación experimental" if support_ok else "insuficiente: stance/swing anulados"
        lines.append(f"• Control de segmentación apoyo/oscilación: izquierda {support_score_l:.0f}/100; derecha {support_score_r:.0f}/100 — {qtmp} (control matemático interno, no escala clínica).")
    coh_flag = val("temporal_coherence_flag")
    coh_err = val("temporal_coherence_max_error_pct")
    if coh_flag is not None and np.isfinite(coh_flag):
        if coh_flag >= 0.5:
            err_txt = f"{coh_err:.1f}%" if coh_err is not None and np.isfinite(coh_err) else "dentro del límite"
            lines.append(f"• Coherencia temporal v0.9.2: SUPERADA — discordancia máxima {err_txt}. Cadencia, asimetría y doble apoyo pueden publicarse si superan además sus controles específicos.")
        else:
            err_txt = f"{coh_err:.1f}%" if coh_err is not None and np.isfinite(coh_err) else "no cuantificable"
            lines.append(f"• Coherencia temporal v0.9.2: NO SUPERADA — discordancia máxima {err_txt}. Cadencia, asimetría temporal y doble apoyo incompatibles se suprimen para evitar falsa precisión.")

    # v0.9.0 · trazabilidad del sujeto cuando existe acompañante/terapeuta.
    manual_keys = ["subject_manual_selection_flag","front_subject_manual_selection_flag","lateral_subject_manual_selection_flag"]
    manual_selected = any((val(k) is not None and np.isfinite(val(k)) and val(k) >= 0.5) for k in manual_keys)
    if manual_selected:
        lines.append("• Selección de sujeto: paciente seleccionado manualmente y seguimiento de identidad bloqueado; las demás personas detectadas quedan excluidas del análisis.")
        cont_candidates = [val("identity_continuity_pct"), val("front_identity_continuity_pct"), val("lateral_identity_continuity_pct")]
        cont_candidates = [float(x) for x in cont_candidates if x is not None and np.isfinite(x)]
        amb_candidates = [val("identity_ambiguous_excluded_pct"), val("front_identity_ambiguous_excluded_pct"), val("lateral_identity_ambiguous_excluded_pct")]
        amb_candidates = [float(x) for x in amb_candidates if x is not None and np.isfinite(x)]
        if cont_candidates:
            lines.append(f"• Continuidad de identidad del sujeto: {min(cont_candidates):.1f}% como valor conservador entre vistas.")
        if amb_candidates:
            lines.append(f"• Frames ambiguos excluidos por riesgo de cambio de identidad: {max(amb_candidates):.1f}% como valor conservador entre vistas.")

    lines.append("")
    lines.append("1. PARÁMETROS ESPACIOTEMPORALES (RITMO, VARIABILIDAD Y CARGA)")
    if np.isfinite(cad):
        lines.append(f"• Cadencia estimada: {cad:.1f} pasos/min ({ref_for(kcad)})")
    if np.isfinite(cv):
        lines.append(f"• Variabilidad temporal (CV): {cv:.1f}% ({ref_for(kcv)})")
    if np.isfinite(asym):
        lines.append(f"• Asimetría temporal global: {asym:.1f}% ({ref_for(kasym)})")

    if np.isfinite(stance_l) and np.isfinite(stance_r):
        if stance_l > stance_r:
            dominant = "IZQUIERDA"
        elif stance_r > stance_l:
            dominant = "DERECHA"
        else:
            dominant = "SIMILAR EN AMBAS EXTREMIDADES"
        sa_txt = f"{stance_asym:.1f}%" if np.isfinite(stance_asym) else "no calculable"
        lines.append("• Tiempo de apoyo 2D estimado (fase de carga):")
        lines.append(f"  - Izquierda: {stance_l:.2f} s ({ref_for(ksl)})")
        lines.append(f"  - Derecha: {stance_r:.2f} s ({ref_for(ksr)})")
        lines.append(f"  - Asimetría de carga: {sa_txt}, con mayor tiempo de apoyo estimado en la extremidad {dominant} ({stance_l:.2f} s vs. {stance_r:.2f} s) ({ref_for(ksa)})")
    else:
        lines.append("• Tiempo de apoyo 2D estimado: NO INFORMADO. La segmentación apoyo/oscilación no supera el control de consistencia temporal en ambos lados; se evita publicar una duración potencialmente falsa (estimación markerless 2D, no plataforma de fuerzas).")

    # Opcional doble apoyo si está disponible, siempre con referencia.
    kds, ds = choose("double_support_pct_2d")
    if np.isfinite(ds):
        lines.append(f"• Apoyo bipodal estimado: {ds:.1f}% del segmento/ciclo analizado (Ref. contextual: en marcha adulta habitual el doble apoyo total suele ocupar aproximadamente 20–24% del ciclo; la estimación v0.9.2 usa el mismo dominio de ciclos rectilíneos que cadencia/apoyo y se anula si falla el control físico interno).")

    lines.append("")
    lines.append("2. CINEMÁTICA PROYECTADA 2D (ESTABILIDAD FRONTAL Y APOYO MONOPODAL)")
    frontal_available = ("Frontal" in (view or "")) or any(str(m.get("key", "")).startswith("front_") for m in metrics)
    if frontal_available:
        kpl, drop_l = choose("trendelenburg_drop_l_deg", prefer_frontal=True)
        kpr, drop_r = choose("trendelenburg_drop_r_deg", prefer_frontal=True)
        kvl, valg_l = choose("dynamic_knee_valgus_l_deg", prefer_frontal=True)
        kvr, valg_r = choose("dynamic_knee_valgus_r_deg", prefer_frontal=True)
        lines.append("• Caída pélvica en suspensión (drop pélvico durante apoyo monopodal):")
        lines.append(f"  - Lado Izquierdo: {drop_l:.1f}° ({ref_for(kpl)})" if np.isfinite(drop_l) else "  - Lado Izquierdo: no calculable (Sin umbral diagnóstico 2D universal; confirmar clínicamente).")
        lines.append(f"  - Lado Derecho: {drop_r:.1f}° ({ref_for(kpr)})" if np.isfinite(drop_r) else "  - Lado Derecho: no calculable (Sin umbral diagnóstico 2D universal; confirmar clínicamente).")
        lines.append("• Valgo dinámico proyectado (desviación medial durante carga):")
        lines.append(f"  - Lado Izquierdo: {valg_l:.1f}° ({ref_for(kvl)})" if np.isfinite(valg_l) else "  - Lado Izquierdo: no calculable (Sin rango normativo validado específicamente para HALPE26 2D).")
        lines.append(f"  - Lado Derecho: {valg_r:.1f}° ({ref_for(kvr)})" if np.isfinite(valg_r) else "  - Lado Derecho: no calculable (Sin rango normativo validado específicamente para HALPE26 2D).")
        lines.append("• Precaución metodológica: el valgo/varo representa únicamente la desviación medial/lateral proyectada en el plano frontal y no una medición 3D multiplanar aislada. El drop pélvico es un descriptor 2D durante apoyo monopodal y debe confirmarse mediante exploración clínica cuando tenga relevancia terapéutica.")
    else:
        lines.append("• La vista actual no permite una estimación frontal fiable de drop pélvico o valgo/varo proyectado (Sin umbral normativo 2D aplicable porque la geometría de captura no es comparable).")

    lines.append("")
    lines.append("3. COORDINACIÓN INTERSEGMENTARIA (TRONCO-PELVIS)")
    if frontal_available:
        kcr, coupling = choose("trunk_pelvis_coupling_r", prefer_frontal=True)
        kph, phase = choose("trunk_pelvis_phase_deg", prefer_frontal=True)
        if np.isfinite(coupling):
            lines.append(f"• Acoplamiento tronco-pelvis (r): {coupling:.2f} r ({ref_for(kcr)})")
        else:
            lines.append("• Acoplamiento tronco-pelvis (r): no calculable (Sin banda normativa universal para este descriptor 2D).")
        if np.isfinite(phase):
            lines.append(f"• Desfase tronco-pelvis: {phase:.1f}° de ciclo ({ref_for(kph)})")
        else:
            lines.append("• Desfase tronco-pelvis: no calculable (Sin banda normativa universal; 360° = 1 ciclo de marcha).")
        lines.append("• Interpretación: r próximo a +1 indica que tronco y pelvis oscilan de forma linealmente similar/en fase; r próximo a −1 sugiere contrafase. El desfase cuantifica el retraso/adelanto relativo en grados de ciclo (360° = 1 ciclo completo) y puede ayudar a describir estrategias compensatorias, pero no diagnostica por sí solo un patrón tipo Duchenne.")
    else:
        lines.append("• La coordinación frontal tronco-pelvis no se estima de forma fiable desde una única vista lateral (Sin referencia normativa aplicable).")

    lines.append("")
    lines.append("4. IMPRESIÓN BIOMECÁNICA GENERAL")
    synthesis = []
    if np.isfinite(cad):
        synthesis.append(f"El registro presenta una cadencia estimada de {cad:.1f} pasos/min, que debe interpretarse frente a la referencia poblacional sana solo como contexto y no como objetivo terapéutico automático")
    if np.isfinite(stance_l) and np.isfinite(stance_r):
        dom = "izquierda" if stance_l > stance_r else "derecha" if stance_r > stance_l else "sin predominio lateral claro"
        synthesis.append(f"la estimación temporal por ciclos continuos muestra mayor permanencia en apoyo en la extremidad {dom} ({stance_l:.2f} s izquierda vs. {stance_r:.2f} s derecha)")
    else:
        synthesis.append("el tiempo de apoyo y su direccionalidad no se informan porque la segmentación temporal no alcanzó consistencia suficiente")
    if frontal_available:
        bits=[]
        if 'drop_l' in locals() and np.isfinite(drop_l): bits.append(f"drop pélvico izquierdo {drop_l:.1f}°")
        if 'drop_r' in locals() and np.isfinite(drop_r): bits.append(f"derecho {drop_r:.1f}°")
        if 'valg_l' in locals() and np.isfinite(valg_l): bits.append(f"valgo proyectado izquierdo {valg_l:.1f}°")
        if 'valg_r' in locals() and np.isfinite(valg_r): bits.append(f"derecho {valg_r:.1f}°")
        if bits:
            synthesis.append("en el plano frontal se observan " + ", ".join(bits) + ", todos ellos descriptores 2D sin umbral diagnóstico universal")
        if 'coupling' in locals() and np.isfinite(coupling):
            ptxt = f" y desfase {phase:.1f}°" if 'phase' in locals() and np.isfinite(phase) else ""
            synthesis.append(f"la coordinación tronco-pelvis muestra r={coupling:.2f}{ptxt}, útil principalmente para seguimiento longitudinal y contextualización clínica")
    if not synthesis:
        lines.append("No hay suficientes métricas válidas para generar una síntesis biomecánica cuantitativa. Debe revisarse la calidad del tracking y el intervalo seleccionado.")
    else:
        # 3–4 líneas cortas, unificadas y sin diagnóstico automático.
        lines.append(". ".join(synthesis) + ".")
        lines.append("Los hallazgos deben integrarse con velocidad de marcha, ayuda técnica, exploración neurológica/musculoesquelética y evolución intraindividual; las métricas markerless 2D no sustituyen una evaluación 3D o instrumentada cuando la decisión clínica depende de fuerzas, contacto o cinemática multiplanar.")

    # La salida principal solicitada es el informe clínico. Mantener una versión paciente separada
    # para compatibilidad con la interfaz anterior, pero no mezclarla dentro de los 4 bloques obligatorios.
    patient=[]
    patient.append("VERSIÓN SIMPLIFICADA PARA EL PACIENTE")
    if np.isfinite(cad):
        patient.append(f"En este registro se estimaron {cad:.1f} pasos por minuto. Las cifras de personas sanas se muestran solo como referencia general; en rehabilitación importa especialmente cómo cambia tu marcha respecto a tus propios registros.")
    if np.isfinite(stance_l) and np.isfinite(stance_r):
        side = "izquierda" if stance_l > stance_r else "derecha" if stance_r > stance_l else "ambas de forma parecida"
        patient.append(f"El análisis sugiere que la pierna que permanece apoyada durante más tiempo es la {side}. La estimación utiliza ciclos continuos y se interpreta junto con la exploración clínica.")
    else:
        patient.append("En este registro no se ofrece un tiempo de apoyo por pierna porque el vídeo no permitió segmentar apoyo y oscilación con suficiente consistencia. Es preferible no mostrar una cifra dudosa.")
    if frontal_available:
        patient.append("También se observa cómo se mantiene la pelvis cuando una pierna está en el aire, cómo se alinean las rodillas de frente y cómo se coordinan el tronco y la pelvis. Son medidas proyectadas en una imagen 2D, útiles sobre todo para comparar tu evolución en sesiones realizadas de forma similar.")
    patient.append("El resultado no se interpreta de forma aislada: se combina con tus síntomas, capacidad funcional, ayudas utilizadas y evolución durante el tratamiento.")
    return "\n".join(lines), "\n\n".join(patient)

# ------------------------- UI -------------------------
# ------------------------- UI -------------------------
st.title("PhysioSentinel Gait")
st.caption(f"Versión {APP_VERSION} · selección multipersona con identidad bloqueada · 2 cámaras · preparación 3D · Supabase tolerante a fallos")

if sb_ready():
    st.success("☁️ Supabase conectado: pacientes, sesiones, métricas y perfiles de calibración pueden persistir. Los vídeos NO se guardan.")
else:
    st.error("Supabase aún no está configurado. Añade SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY en Streamlit Secrets.")

# Estado base
for k, v in {
    "pose_done": False, "metrics_done": False, "temp_deleted": False,
    "analysis_df": None, "analysis_df2": None,
    "annotated_video_bytes": None, "annotated_video2_bytes": None,
    "sync_offset_auto_s": 0.0, "sync_offset_user_s": 0.0,
    "sync_correlation": np.nan, "sync_quality": "No calculable",
    "camera1_quality": None, "camera2_quality": None,
    "pose_raw_done": False, "subject_locked": False,
    "subject_selection_cam1": None, "subject_selection_cam2": None,
    "subject_preview_cam1": None, "subject_preview_cam2": None,
    "tracking_info_cam1": None, "tracking_info_cam2": None,
    "pose_json1": None, "pose_json2": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Perfiles de calibración disponibles
calibration_names = []
if sb_ready():
    try:
        calibration_names = [x["name"] for x in sb_list_calibrations()]
    except Exception:
        calibration_names = []

with st.sidebar:
    st.header("Sesión")
    patient = st.text_input("Paciente / código", value="Prueba")
    record = st.text_input("Nombre del registro", value="Marcha")
    st.caption("Usa un código seudonimizado, no nombre y apellidos.")
    st.divider()
    mode = st.radio(
        "Modo de análisis",
        ["1 cámara · 2D", "2 cámaras · frontal/posterior + lateral · preparación 3D"],
        index=0,
    )
    is_two_cam = mode.startswith("2 cámaras")
    view = st.radio("Vista", ["Frontal/posterior", "Lateral"], index=0) if not is_two_cam else "Frontal+Lateral"
    frontal_orientation = st.selectbox(
        "Sentido de la toma frontal/posterior",
        ["No especificada", "Frontal", "Posterior", "Mixta/ida-vuelta"],
        index=0,
    ) if "Frontal" in view else "No aplica"
    assistive_device = st.selectbox("Ayuda técnica", ASSISTIVE_OPTIONS, index=0)
    if assistive_device != "Sin ayuda":
        st.caption("La app medirá la visibilidad por regiones y marcará métricas condicionadas por posibles oclusiones.")
    subject_mode = st.radio(
        "Sujeto biomecánico",
        ["Automático · una sola persona", "Selección manual · multipersona"],
        index=0,
        help="Usa selección manual cuando el paciente camina acompañado, asistido o estrechamente supervisado."
    )
    manual_subject_mode = subject_mode.startswith("Selección manual")
    if manual_subject_mode:
        st.caption("v0.9.1: primero se detectan las personas; tú eliges al paciente y después la identidad queda bloqueada. Los frames ambiguos se excluyen.")
    if is_two_cam:
        opts = ["Sin calibración"] + calibration_names
        default_cal = st.session_state.get("selected_calibration_name") or "Sin calibración"
        try:
            idx = opts.index(default_cal)
        except ValueError:
            idx = 0
        selected_calibration = st.selectbox("Perfil de calibración 3D", opts, index=idx)
        st.session_state.selected_calibration_name = None if selected_calibration == "Sin calibración" else selected_calibration
    else:
        selected_calibration = "Sin calibración"
        st.session_state.selected_calibration_name = None
    st.divider()
    st.markdown("**Escala espacial frontal (opcional)**")
    scale_cm_per_px = st.number_input("cm por píxel", min_value=0.0, value=float(st.session_state.get("scale_cm_per_px",0.0)), step=0.001, format="%.4f", help="Solo si dispones de una calibración espacial válida en el plano de marcha. 0 = sin escala métrica.")
    st.session_state.scale_cm_per_px=float(scale_cm_per_px)
    st.caption("Sin escala válida, CoM y BoS se mantienen como proxies relativos/píxel y no se convierten a cm.")
    st.divider()
    st.markdown("**Motor interno**")
    st.write("Pose2Sim + RTMPose")
    st.write("Body_with_feet / HALPE26")
    st.caption("Vídeos → /tmp → pose 2D → sincronización → resultados → Supabase → eliminación")


tabs = st.tabs([
    "1 · Vídeos", "2 · Calidad", "3 · Analizar marcha", "4 · Resultados 2D",
    "5 · Pacientes / Evolución", "6 · Redacción informe", "7 · 3D / Calibración"
])

with tabs[0]:
    st.subheader("Carga temporal de vídeo")
    st.info("Los vídeos se usan únicamente para este análisis. No se suben a Supabase ni quedan guardados en el histórico.")
    if not is_two_cam:
        up1 = st.file_uploader(f"Vídeo {view.lower()}", type=["mp4","mov","avi","mkv"], key="uploader_video1")
        if up1:
            st.video(up1)
        up2 = None
    else:
        st.markdown("### Protocolo 2 cámaras")
        st.write("Cámara 1: frontal/posterior. Cámara 2: lateral. Ambas deben grabar simultáneamente la misma zona de marcha.")
        st.caption("Para facilitar la sincronización, realiza al inicio un evento corporal breve y visible en ambas cámaras (por ejemplo, una elevación vertical rápida) antes de iniciar la marcha.")
        c1, c2 = st.columns(2)
        with c1:
            up1 = st.file_uploader("Cámara 1 · frontal/posterior", type=["mp4","mov","avi","mkv"], key="uploader_front")
            if up1:
                st.video(up1)
        with c2:
            up2 = st.file_uploader("Cámara 2 · lateral", type=["mp4","mov","avi","mkv"], key="uploader_side")
            if up2:
                st.video(up2)
        st.caption("v0.7 procesa RTMPose en las dos cámaras, calcula 2D complementario por vista y estima el desfase temporal. La triangulación 3D todavía no se ejecuta automáticamente.")

    if st.button("Crear sesión temporal", type="primary", use_container_width=True):
        if up1 is None or (is_two_cam and up2 is None):
            st.error("Selecciona el/los vídeo(s) necesarios.")
        else:
            try:
                folder = create_temp_session(patient, record)
                p1 = folder / "videos" / f"cam01{Path(up1.name).suffix.lower() or '.mp4'}"
                save_upload(up1, p1)
                meta1 = video_metadata(p1)
                if not meta1:
                    raise RuntimeError("No puedo leer el vídeo de cámara 1.")
                p2 = None
                meta2 = None
                if up2 is not None:
                    p2 = folder / "videos" / f"cam02{Path(up2.name).suffix.lower() or '.mp4'}"
                    save_upload(up2, p2)
                    meta2 = video_metadata(p2)
                    if not meta2:
                        raise RuntimeError("No puedo leer el vídeo de cámara 2.")
                st.session_state.update({
                    "session_dir": str(folder), "video1_path": str(p1), "video2_path": str(p2) if p2 else None,
                    "meta1": meta1, "meta2": meta2, "mode": mode, "view": view,
                    "subject_mode": subject_mode,
                    "pose_done": False, "pose_raw_done": False, "subject_locked": False,
                    "subject_selection_cam1": None, "subject_selection_cam2": None,
                    "subject_preview_cam1": None, "subject_preview_cam2": None,
                    "tracking_info_cam1": None, "tracking_info_cam2": None,
                    "metrics_done": False, "temp_deleted": False,
                    "analysis_df": None, "analysis_df2": None,
                    "annotated_video_bytes": None, "annotated_video2_bytes": None,
                    "patient_code": patient.strip(), "record_name": record.strip(),
                    "assistive_device": assistive_device, "frontal_orientation": frontal_orientation,
                    "sync_offset_auto_s": 0.0, "sync_offset_user_s": 0.0,
                    "sync_correlation": np.nan, "sync_quality": "No calculable",
                    "camera1_quality": None, "camera2_quality": None,
                    "cloud_session_id": None,
                    "selected_calibration_name": st.session_state.get("selected_calibration_name"),
                })
                cloud_ok = False
                if sb_ready():
                    try:
                        sid = sb_create_session(
                            patient.strip(), record.strip(), mode, view, meta1,
                            assistive_device, frontal_orientation,
                            meta2=meta2,
                            calibration_profile_name=st.session_state.get("selected_calibration_name"),
                        )
                        st.session_state["cloud_session_id"] = sid
                        cloud_ok = True
                    except Exception as sb_e:
                        st.session_state["cloud_session_id"] = None
                        st.warning(f"La sesión temporal se ha creado, pero Supabase no pudo registrar esta sesión: {sb_e}")
                if cloud_ok:
                    st.success("Sesión temporal creada. Supabase conserva la sesión y su contexto; los vídeos permanecen solo en /tmp.")
                else:
                    st.success("Sesión temporal creada. Puedes analizar y ver resultados aunque Supabase no esté disponible.")
            except Exception as e:
                st.error(str(e))

with tabs[1]:
    st.subheader("Control de calidad de captura")
    meta1 = st.session_state.get("meta1")
    meta2 = st.session_state.get("meta2")
    if not meta1:
        st.info("Crea primero una sesión temporal.")
    else:
        if st.session_state.get("mode", mode).startswith("2 cámaras") and meta2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Cámara 1 · frontal/posterior")
                st.metric("FPS", f"{meta1['fps']:.2f}")
                st.metric("Duración", f"{meta1['duration']:.2f} s")
                st.write(f"Resolución: **{meta1['width']} × {meta1['height']}**")
            with c2:
                st.markdown("### Cámara 2 · lateral")
                st.metric("FPS", f"{meta2['fps']:.2f}")
                st.metric("Duración", f"{meta2['duration']:.2f} s")
                st.write(f"Resolución: **{meta2['width']} × {meta2['height']}**")
            fps_diff = abs(meta1["fps"] - meta2["fps"])
            dur_diff = abs(meta1["duration"] - meta2["duration"])
            a, b = st.columns(2)
            a.metric("Diferencia FPS", f"{fps_diff:.3f}")
            b.metric("Diferencia de duración", f"{dur_diff:.2f} s")
            if fps_diff <= 0.5:
                st.success("Las frecuencias de imagen son suficientemente próximas para la preparación 3D.")
            else:
                st.warning("Las cámaras tienen FPS diferentes. Se puede analizar 2D, pero la reconstrucción 3D requerirá remuestreo/sincronización cuidadosa.")
            if dur_diff > 2.0:
                st.warning("La duración difiere más de 2 s. Revisa que ambas grabaciones cubran el mismo ensayo.")
        else:
            a,b,c,d = st.columns(4)
            a.metric("FPS", f"{meta1['fps']:.1f}")
            b.metric("Duración", f"{meta1['duration']:.1f} s")
            c.metric("Resolución", f"{meta1['width']} × {meta1['height']}")
            d.metric("Orientación", meta1['orientation'])
        if st.session_state.get("assistive_device", "Sin ayuda") != "Sin ayuda":
            st.info(f"Marcha con ayuda técnica: **{st.session_state.get('assistive_device')}**. Tras RTMPose se calculará la visibilidad por cámara y región corporal.")

with tabs[2]:
    st.subheader("Analizar marcha")
    if not st.session_state.get("session_dir"):
        st.info("Crea primero una sesión temporal.")
    elif st.session_state.get("temp_deleted"):
        st.info("Los archivos temporales ya fueron eliminados después de guardar los resultados.")
    else:
        current_two = st.session_state.get("mode", "").startswith("2 cámaras")
        manual_mode = str(st.session_state.get("subject_mode","")).startswith("Selección manual")
        st.write("Motor: **Pose2Sim + RTMPose · Body_with_feet (HALPE26)**")
        if manual_mode:
            st.info("**Modo multipersona v0.9.1:** la app detectará las personas, pero NO decidirá cuál es el paciente. Después de la detección deberás seleccionarlo explícitamente.")
        if current_two:
            st.write("Se realizará detección de pose en **cam01 frontal/posterior y cam02 lateral** y después se estimará su desfase temporal.")

        if not st.session_state.get("pose_raw_done") and st.button("▶ Detectar pose" if manual_mode else "▶ Analizar marcha", type="primary", use_container_width=True):
            try:
                session_dir = Path(st.session_state.session_dir)
                with st.spinner("Detectando pose con Pose2Sim/RTMPose. En 2 cámaras el proceso tarda aproximadamente el doble que una sola vista..."):
                    cfg = prepare_config(session_dir)
                    run_pose2sim(cfg)
                json1 = find_pose_json_dir(session_dir, "cam01")
                if not json1:
                    raise RuntimeError("Pose2Sim terminó pero no encuentro los JSON de cam01.")
                st.session_state.pose_json1 = str(json1)

                json2 = None
                if current_two:
                    json2 = find_pose_json_dir(session_dir, "cam02")
                    if not json2:
                        raise RuntimeError("Pose2Sim terminó pero no encuentro los JSON de cam02.")
                    st.session_state.pose_json2 = str(json2)

                if manual_mode:
                    sel1 = scan_subject_candidates(json1)
                    if not sel1 or not sel1.get("candidates"):
                        raise RuntimeError("No encuentro ninguna persona suficientemente visible en cam01 para realizar la selección manual.")
                    st.session_state.subject_selection_cam1 = sel1
                    st.session_state.subject_preview_cam1 = render_subject_preview(st.session_state.video1_path, sel1)

                    if current_two and json2:
                        sel2 = scan_subject_candidates(json2)
                        if not sel2 or not sel2.get("candidates"):
                            raise RuntimeError("No encuentro ninguna persona suficientemente visible en cam02 para realizar la selección manual.")
                        st.session_state.subject_selection_cam2 = sel2
                        st.session_state.subject_preview_cam2 = render_subject_preview(st.session_state.video2_path, sel2)

                    st.session_state.pose_raw_done = True
                    st.session_state.pose_done = False
                    st.success("Pose detectada. Selecciona ahora el paciente en la imagen y bloquea su identidad.")
                else:
                    df1 = load_pose_dataframe(json1)
                    if df1.empty:
                        raise RuntimeError("No se pudieron leer keypoints HALPE26 de cam01.")
                    st.session_state.analysis_df = df1
                    q1 = camera_pose_quality(df1)
                    st.session_state.camera1_quality = q1

                    if current_two:
                        df2 = load_pose_dataframe(json2)
                        if df2.empty:
                            raise RuntimeError("No se pudieron leer keypoints HALPE26 de cam02.")
                        st.session_state.analysis_df2 = df2
                        q2 = camera_pose_quality(df2)
                        st.session_state.camera2_quality = q2
                        meta1 = st.session_state.meta1; meta2 = st.session_state.meta2
                        off, corr, sq = estimate_sync_offset(df1, meta1["fps"], df2, meta2["fps"], max_offset_s=2.0)
                        st.session_state.sync_offset_auto_s = off
                        st.session_state.sync_offset_user_s = off
                        st.session_state.sync_correlation = corr
                        st.session_state.sync_quality = sq
                        ready, reasons = readiness_3d(st.session_state.mode, q1, q2, sq, st.session_state.get("selected_calibration_name"))
                        if sb_ready() and st.session_state.get("cloud_session_id"):
                            sb_update_session_3d(
                                st.session_state.cloud_session_id,
                                sync_offset_s=off,
                                sync_correlation=corr if np.isfinite(corr) else None,
                                sync_quality=sq,
                                calibration_profile_name=st.session_state.get("selected_calibration_name"),
                                ready_3d=bool(ready),
                            )
                        st.success(f"Pose completada en dos cámaras: cam01 {len(df1)} frames útiles · cam02 {len(df2)} frames útiles.")
                        st.info(f"Sincronización automática experimental: **{off:+.3f} s** · correlación **{corr:.2f}** · calidad **{sq}**.")
                    else:
                        st.session_state.analysis_df2 = None
                        st.success(f"Pose completada: {len(df1)} frames útiles.")
                    st.session_state.pose_raw_done = True
                    st.session_state.pose_done = True
                    st.session_state.subject_locked = True
                    st.info("Abre **4 · Resultados 2D**, revisa el desfase/segmento válido y calcula. Después se eliminarán los vídeos temporales.")
            except Exception as e:
                st.error(f"Error durante el análisis: {e}")
                with st.expander("Detalles técnicos"):
                    st.code(traceback.format_exc())

        # Segunda etapa exclusivamente multipersona: elección humana + bloqueo de identidad.
        if manual_mode and st.session_state.get("pose_raw_done") and not st.session_state.get("subject_locked"):
            st.markdown("### Selección manual del sujeto biomecánico")
            st.warning("Selecciona al **paciente**, no al terapeuta/acompañante. La app conservará esa identidad y rechazará frames ambiguos en lugar de cambiar de persona.")

            sel1=st.session_state.get("subject_selection_cam1")
            if sel1:
                st.markdown("#### Cámara 1")
                if st.session_state.get("subject_preview_cam1"):
                    st.image(st.session_state.subject_preview_cam1, caption=f"Frame de selección {sel1['frame']} · sujetos ordenados de izquierda a derecha", use_container_width=True)
                labels1=[c["label"] for c in sel1["candidates"]]
                choice1=st.radio("Paciente en cámara 1", labels1, key="manual_subject_choice_cam1", horizontal=True)
            else:
                choice1=None

            choice2=None
            sel2=st.session_state.get("subject_selection_cam2")
            if current_two and sel2:
                st.markdown("#### Cámara 2")
                if st.session_state.get("subject_preview_cam2"):
                    st.image(st.session_state.subject_preview_cam2, caption=f"Frame de selección {sel2['frame']} · selecciona a la misma persona física", use_container_width=True)
                labels2=[c["label"] for c in sel2["candidates"]]
                choice2=st.radio("Paciente en cámara 2", labels2, key="manual_subject_choice_cam2", horizontal=True)

            if st.button("🔒 Bloquear sujeto y preparar análisis biomecánico", type="primary", use_container_width=True):
                try:
                    c1=next(c for c in sel1["candidates"] if c["label"]==choice1)
                    df1,t1=load_pose_dataframe_tracked(Path(st.session_state.pose_json1), sel1["frame"], c1["person_index"])
                    if df1.empty or len(df1)<10:
                        raise RuntimeError("El seguimiento del sujeto seleccionado en cam01 no produce suficientes frames fiables.")
                    st.session_state.analysis_df=df1
                    st.session_state.tracking_info_cam1=t1
                    q1=camera_pose_quality(df1)
                    st.session_state.camera1_quality=q1

                    if current_two:
                        c2=next(c for c in sel2["candidates"] if c["label"]==choice2)
                        df2,t2=load_pose_dataframe_tracked(Path(st.session_state.pose_json2), sel2["frame"], c2["person_index"])
                        if df2.empty or len(df2)<10:
                            raise RuntimeError("El seguimiento del sujeto seleccionado en cam02 no produce suficientes frames fiables.")
                        st.session_state.analysis_df2=df2
                        st.session_state.tracking_info_cam2=t2
                        q2=camera_pose_quality(df2)
                        st.session_state.camera2_quality=q2
                        meta1=st.session_state.meta1; meta2=st.session_state.meta2
                        off,corr,sq=estimate_sync_offset(df1,meta1["fps"],df2,meta2["fps"],max_offset_s=2.0)
                        st.session_state.sync_offset_auto_s=off
                        st.session_state.sync_offset_user_s=off
                        st.session_state.sync_correlation=corr
                        st.session_state.sync_quality=sq
                        ready,reasons=readiness_3d(st.session_state.mode,q1,q2,sq,st.session_state.get("selected_calibration_name"))
                        if sb_ready() and st.session_state.get("cloud_session_id"):
                            sb_update_session_3d(
                                st.session_state.cloud_session_id,
                                sync_offset_s=off,
                                sync_correlation=corr if np.isfinite(corr) else None,
                                sync_quality=sq,
                                calibration_profile_name=st.session_state.get("selected_calibration_name"),
                                ready_3d=bool(ready),
                            )
                    else:
                        st.session_state.analysis_df2=None

                    st.session_state.subject_locked=True
                    st.session_state.pose_done=True
                    st.success("🔒 Sujeto biomecánico bloqueado. La otra persona queda excluida del análisis.")
                    st.write(
                        f"Continuidad de identidad cam01: **{t1.get('identity_continuity_pct',np.nan):.1f}%** · "
                        f"frames excluidos: **{t1.get('frames_excluded',0)}** · calidad: **{t1.get('quality','')}**."
                    )
                    if current_two:
                        st.write(
                            f"Continuidad de identidad cam02: **{t2.get('identity_continuity_pct',np.nan):.1f}%** · "
                            f"frames excluidos: **{t2.get('frames_excluded',0)}** · calidad: **{t2.get('quality','')}**."
                        )
                    st.info("Abre **4 · Resultados 2D** y calcula el intervalo válido. Las métricas se obtendrán únicamente del sujeto seleccionado.")
                except Exception as e:
                    st.error(f"No se pudo bloquear el sujeto: {e}")
                    with st.expander("Detalles técnicos"):
                        st.code(traceback.format_exc())

        elif manual_mode and st.session_state.get("subject_locked"):
            t1=st.session_state.get("tracking_info_cam1") or {}
            st.success("🔒 Sujeto biomecánico bloqueado.")
            if t1:
                st.caption(
                    f"Cam01 · continuidad identidad {t1.get('identity_continuity_pct',np.nan):.1f}% · "
                    f"excluidos {t1.get('frames_excluded',0)}/{t1.get('frames_total',0)} frames · "
                    f"cambios de identidad evitados {t1.get('switches_prevented',0)}."
                )

with tabs[3]:
    st.subheader("Resultados 2D complementarios")
    df1 = st.session_state.get("analysis_df")
    df2 = st.session_state.get("analysis_df2")
    meta1 = st.session_state.get("meta1")
    meta2 = st.session_state.get("meta2")
    current_two = st.session_state.get("mode", "").startswith("2 cámaras")

    if (df1 is None or not st.session_state.get("pose_done") or not meta1) and not st.session_state.get("metrics_done"):
        st.info("Ejecuta primero **Analizar marcha**.")

    if df1 is not None and meta1:
        if current_two and df2 is not None and meta2:
            st.markdown("### Sincronización de cámaras")
            auto = float(st.session_state.get("sync_offset_auto_s", 0.0))
            corr = st.session_state.get("sync_correlation", np.nan)
            sq = st.session_state.get("sync_quality", "No calculable")
            st.caption("Convención: un valor positivo significa que el mismo evento aparece más tarde en cam02; para alinear, cam02 se avanza/recorta ese tiempo.")
            st.write(f"Estimación automática: **{auto:+.3f} s** · correlación **{fmt(corr,2)}** · calidad **{sq}**")
            max_sync = 2.0
            sync_user = st.number_input(
                "Desfase cam02 respecto a cam01 (s)",
                min_value=-max_sync, max_value=max_sync,
                value=float(st.session_state.get("sync_offset_user_s", auto)),
                step=0.01,
                help="Puedes corregir manualmente el desfase si conoces el evento de sincronización.",
            )
            st.session_state.sync_offset_user_s = float(sync_user)
            dur1, dur2 = float(meta1["duration"]), float(meta2["duration"])
            common_start = max(0.0, -float(sync_user))
            common_end = min(dur1, dur2 - float(sync_user))
            if common_end <= common_start + 1.0:
                st.error("No queda un intervalo temporal común suficiente con el desfase seleccionado.")
            else:
                start_s, end_s = st.slider(
                    "Intervalo válido común (tiempo de cam01)",
                    float(round(common_start,2)), float(round(common_end,2)),
                    (float(round(common_start,2)), float(round(common_end,2))),
                    step=0.02,
                )
                st.caption(f"Cam02 analizará aproximadamente {start_s+sync_user:.2f}–{end_s+sync_user:.2f} s para representar el mismo intervalo físico.")
                if st.button("Calcular ambas vistas, guardar histórico y eliminar vídeos", type="primary", use_container_width=True):
                    try:
                        f1a, f1b = int(round(start_s*meta1["fps"])), int(round(end_s*meta1["fps"]))
                        s2a, s2b = start_s + sync_user, end_s + sync_user
                        f2a, f2b = int(round(s2a*meta2["fps"])), int(round(s2b*meta2["fps"]))
                        m_front, chart_front, seg_front = compute_metrics(
                            df1, float(meta1["fps"]), f1a, f1b, "Frontal/posterior", st.session_state.get("assistive_device","Sin ayuda"), st.session_state.get("scale_cm_per_px",0.0)
                        )
                        m_lat, chart_lat, seg_lat = compute_metrics(
                            df2, float(meta2["fps"]), f2a, f2b, "Lateral", st.session_state.get("assistive_device","Sin ayuda"), 0.0
                        )
                        metrics = prefix_metrics(m_front, "front", "Frontal/posterior") + prefix_metrics(m_lat, "lateral", "Lateral")
                        metrics += tracking_metrics(st.session_state.get("tracking_info_cam1"), "front", "Frontal/posterior")
                        metrics += tracking_metrics(st.session_state.get("tracking_info_cam2"), "lateral", "Lateral")
                        metrics += [
                            {"key":"sync_offset_cam02_s","label":"Desfase cam02 vs cam01","value":sync_user,"unit":"s","quality":sq,"notes":"Sincronización temporal experimental; valor positivo = evento más tardío en cam02."},
                            {"key":"sync_correlation","label":"Correlación de sincronización automática","value":corr,"unit":"r","quality":sq,"notes":"Correlación heurística de movimiento corporal vertical entre cámaras."},
                        ]
                        q1 = st.session_state.get("camera1_quality") or camera_pose_quality(df1)
                        q2 = st.session_state.get("camera2_quality") or camera_pose_quality(df2)
                        ready, reasons = readiness_3d(st.session_state.mode, q1, q2, sq, st.session_state.get("selected_calibration_name"))
                        metrics.append({"key":"ready_3d_flag","label":"Preparación para triangulación 3D","value":1.0 if ready else 0.0,"unit":"bool","quality":"Preparado" if ready else "Pendiente","notes":"; ".join(reasons) if reasons else "Dos cámaras, visibilidad suficiente, sincronización aceptable y perfil de calibración seleccionado."})
                        st.session_state.metrics = metrics
                        st.session_state.chart_front = chart_front
                        st.session_state.chart_lateral = chart_lat
                        st.session_state.metrics_done = True
                        st.session_state.ready_3d = ready
                        st.session_state.ready_3d_reasons = reasons
                        cloud_saved = False
                        if sb_ready() and st.session_state.get("cloud_session_id"):
                            try:
                                sb_save_metrics(st.session_state.cloud_session_id, metrics, start_s, end_s)
                                sb_update_session_3d(
                                    st.session_state.cloud_session_id,
                                    sync_offset_s=float(sync_user),
                                    sync_correlation=float(corr) if np.isfinite(corr) else None,
                                    sync_quality=sq,
                                    calibration_profile_name=st.session_state.get("selected_calibration_name"),
                                    ready_3d=bool(ready),
                                )
                                cloud_saved = True
                            except Exception as sb_e:
                                st.warning(f"Resultados calculados, pero no se pudieron guardar en Supabase: {sb_e}")
                        session_dir = Path(st.session_state.session_dir)
                        try:
                            out1 = session_dir / "gait_front_web.mp4"
                            made1 = render_angle_video(Path(st.session_state.video1_path), df1, out1, "Frontal/posterior", st.session_state.get("assistive_device","Sin ayuda"))
                            if made1 and made1.exists():
                                st.session_state.annotated_video_bytes = made1.read_bytes()
                            out2 = session_dir / "gait_lateral_web.mp4"
                            made2 = render_angle_video(Path(st.session_state.video2_path), df2, out2, "Lateral", st.session_state.get("assistive_device","Sin ayuda"))
                            if made2 and made2.exists():
                                st.session_state.annotated_video2_bytes = made2.read_bytes()
                        except Exception:
                            st.session_state.annotated_video_bytes = None
                            st.session_state.annotated_video2_bytes = None
                        cleanup_temp_session(session_dir)
                        st.session_state.temp_deleted = True
                        st.session_state.analysis_df = None
                        st.session_state.analysis_df2 = None
                        st.success("✅ Resultados calculados y mostrados. ✅ Vídeos y archivos Pose2Sim eliminados del servidor temporal." + (" ✅ Histórico guardado en Supabase." if cloud_saved else " ⚠️ Histórico no guardado en Supabase."))
                    except Exception as e:
                        st.error(str(e))
                        with st.expander("Detalles técnicos"):
                            st.code(traceback.format_exc())
        else:
            fps = float(meta1["fps"])
            duration = (int(df1.frame.max()) + 1) / fps
            start_s, end_s = st.slider(
                "Intervalo válido (segundos)", 0.0, float(round(duration,2)),
                (0.0,float(round(duration,2))), step=max(0.01, round(1/fps,2))
            )
            if st.button("Calcular, guardar histórico y eliminar vídeo", type="primary", use_container_width=True):
                try:
                    start_frame = int(round(start_s*fps)); end_frame = int(round(end_s*fps))
                    metrics, chart, seg = compute_metrics(df1, fps, start_frame, end_frame, st.session_state.get("view",""), st.session_state.get("assistive_device","Sin ayuda"), st.session_state.get("scale_cm_per_px",0.0))
                    metrics += tracking_metrics(st.session_state.get("tracking_info_cam1"), "", st.session_state.get("view",""))
                    st.session_state.metrics = metrics
                    st.session_state.chart = chart
                    st.session_state.metrics_done = True
                    cloud_saved = False
                    if sb_ready() and st.session_state.get("cloud_session_id"):
                        try:
                            sb_save_metrics(st.session_state.cloud_session_id, metrics, start_s, end_s)
                            cloud_saved = True
                        except Exception as sb_e:
                            st.warning(f"Resultados calculados, pero no se pudieron guardar en Supabase: {sb_e}")
                    session_dir = Path(st.session_state.session_dir)
                    try:
                        out = session_dir / "gait_angles_web.mp4"
                        made = render_angle_video(Path(st.session_state.video1_path), df1, out, st.session_state.get("view",""), st.session_state.get("assistive_device","Sin ayuda"))
                        if made and made.exists():
                            st.session_state.annotated_video_bytes = made.read_bytes()
                    except Exception:
                        st.session_state.annotated_video_bytes = None
                    cleanup_temp_session(session_dir)
                    st.session_state.temp_deleted = True
                    st.session_state.analysis_df = None
                    st.success("✅ Resultados calculados y mostrados. ✅ Vídeo y archivos Pose2Sim eliminados del servidor temporal." + (" ✅ Histórico guardado en Supabase." if cloud_saved else " ⚠️ Histórico no guardado en Supabase."))
                except Exception as e:
                    st.error(str(e))
                    with st.expander("Detalles técnicos"):
                        st.code(traceback.format_exc())

    if st.session_state.get("metrics_done"):
        metrics = st.session_state.metrics
        st.markdown("### Resumen")
        if st.session_state.get("mode", "").startswith("2 cámaras"):
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Cadencia lateral", fmt(metric_value(metrics,"lateral_cadence_exp"),1)+" pasos/min")
            c2.metric("Tracking frontal", fmt(metric_value(metrics,"front_good_frames_pct"),1)+" %")
            c3.metric("Tracking lateral", fmt(metric_value(metrics,"lateral_good_frames_pct"),1)+" %")
            c4.metric("Desfase cam02", fmt(metric_value(metrics,"sync_offset_cam02_s"),2)+" s")
            st.caption(f"Cadencia lateral ({reference_text_for_metric('lateral_cadence_exp')})")
            sl=metric_value(metrics,"lateral_stance_time_l_2d"); sr=metric_value(metrics,"lateral_stance_time_r_2d"); sa=metric_value(metrics,"lateral_stance_asymmetry_2d")
            qsl=metric_value(metrics,"lateral_support_segmentation_score_l"); qsr=metric_value(metrics,"lateral_support_segmentation_score_r")
            if sl is not None and sr is not None and np.isfinite(sl) and np.isfinite(sr):
                side="IZQUIERDA" if sl>sr else ("DERECHA" if sr>sl else "SIMILAR")
                st.info(f"Apoyo 2D por ciclos continuos (lateral): I {sl:.2f} s · D {sr:.2f} s · asimetría {fmt(sa,1)} % · mayor tiempo estimado: **{side}**. ({reference_text_for_metric('lateral_stance_asymmetry_2d')})")
            else:
                st.warning(f"Tiempo de apoyo lateral NO informado: la segmentación apoyo/oscilación no superó el control temporal (I {fmt(qsl,0)}/100 · D {fmt(qsr,0)}/100).")
            st.markdown("#### Consistencia temporal · cámara lateral")
            t1,t2,t3,t4 = st.columns(4)
            t1.metric("Eventos detectados", fmt(metric_value(metrics,"lateral_step_events_detected"),0))
            t2.metric("Duración segmento", fmt(metric_value(metrics,"lateral_segment_duration_s"),2)+" s")
            t3.metric("Eventos esperados", fmt(metric_value(metrics,"lateral_expected_steps_from_cadence"),1))
            t4.metric("Discrepancia", fmt(metric_value(metrics,"lateral_step_count_consistency_error_pct"),1)+" %")
            cq = metric_quality(metrics,"lateral_step_count_consistency_error_pct") or "No calculable"
            if cq == "Alta": st.success(f"Consistencia interna: {cq}")
            elif cq == "Moderada": st.info(f"Consistencia interna: {cq}")
            else: st.warning(f"Consistencia interna: {cq}. Conviene revisar visualmente los eventos detectados.")
            st.caption("Control interno: número de eventos de alternancia detectados ↔ cadencia estimada ↔ duración. Los eventos siguen siendo experimentales y no equivalen todavía a heel-strikes validados.")
            st.markdown("### Resumen biomecánico")
            for paragraph in biomech_summary(metrics, "Lateral", prefix="lateral_"):
                st.write(paragraph)
            for paragraph in biomech_summary(metrics, "Frontal/posterior", prefix="front_")[1:]:
                st.write(paragraph)
            ready = bool(st.session_state.get("ready_3d", False))
            if ready:
                st.success("🧭 Sesión preparada para una futura triangulación 3D: dos poses, sincronización aceptable, visibilidad suficiente y calibración seleccionada.")
            else:
                st.warning("La sesión todavía no cumple todos los requisitos de preparación 3D.")
                for reason in st.session_state.get("ready_3d_reasons", []):
                    st.write(f"• {reason}")
            st.warning("La v0.7 **no triangula todavía coordenadas 3D**. Los resultados siguientes continúan siendo 2D, pero cada plano se interpreta desde la cámara apropiada.")
            chart_front = st.session_state.get("chart_front")
            chart_lat = st.session_state.get("chart_lateral")
            if chart_front is not None:
                st.markdown("### Cámara 1 · frontal/posterior")
                for title, cols in [
                    ("Pelvis y tronco", ["Oblicuidad pélvica","Inclinación lateral del tronco","Relación hombros-pelvis"]),
                    ("Hombros", ["Oblicuidad de hombros"]),
                    ("Cadera / rodilla · eje frontal", ["Rodilla frontal izquierda","Rodilla frontal derecha"]),
                    ("Pie · orientación distal", ["Orientación pie izquierda","Orientación pie derecha"]),
                    ("Pie · inclinación del retropié", ["Retropié izquierda","Retropié derecha"]),
                    ("Avanzado frontal", ["CoM proxy lateral (px)","Valgo dinámico proyectado I","Valgo dinámico proyectado D"]),
                ]:
                    with st.expander(title, expanded=(title == "Pelvis y tronco")):
                        st.line_chart(chart_front.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo proyectado (°)")
            if chart_lat is not None:
                st.markdown("### Cámara 2 · lateral")
                for title, cols in [
                    ("Rodillas", ["Rodilla izquierda","Rodilla derecha"]),
                    ("Caderas", ["Cadera izquierda","Cadera derecha"]),
                    ("Tobillos / pie", ["Tobillo izquierda","Tobillo derecha"]),
                    ("Hombros", ["Hombro izquierda","Hombro derecha"]),
                ]:
                    with st.expander(title, expanded=(title == "Rodillas")):
                        st.line_chart(chart_lat.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo 2D (°)")
            if st.session_state.get("annotated_video_bytes") or st.session_state.get("annotated_video2_bytes"):
                st.markdown("### Vídeos con esqueleto + métricas")
                v1,v2 = st.columns(2)
                with v1:
                    if st.session_state.get("annotated_video_bytes"):
                        st.caption("Cámara 1 · frontal/posterior")
                        st.video(st.session_state.annotated_video_bytes)
                with v2:
                    if st.session_state.get("annotated_video2_bytes"):
                        st.caption("Cámara 2 · lateral")
                        st.video(st.session_state.annotated_video2_bytes)
        else:
            chart = st.session_state.chart
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Cadencia estimada", fmt(metric_value(metrics,"cadence_exp"),1)+" pasos/min")
            c2.metric("Regularidad temporal", fmt(metric_value(metrics,"regularity_cv"),1)+" % CV")
            c3.metric("Tracking válido", fmt(metric_value(metrics,"good_frames_pct"),1)+" %")
            c4.metric("Asimetría temporal", fmt(metric_value(metrics,"temporal_asymmetry_exp"),1)+" %")
            st.caption(f"Cadencia ({reference_text_for_metric('cadence_exp')}) · Variabilidad ({reference_text_for_metric('regularity_cv')}) · Asimetría ({reference_text_for_metric('temporal_asymmetry_exp')})")
            sl=metric_value(metrics,"stance_time_l_2d"); sr=metric_value(metrics,"stance_time_r_2d"); sa=metric_value(metrics,"stance_asymmetry_2d")
            qsl=metric_value(metrics,"support_segmentation_score_l"); qsr=metric_value(metrics,"support_segmentation_score_r")
            if sl is not None and sr is not None and np.isfinite(sl) and np.isfinite(sr):
                side="IZQUIERDA" if sl>sr else ("DERECHA" if sr>sl else "SIMILAR")
                st.info(f"Apoyo 2D por ciclos continuos: I {sl:.2f} s · D {sr:.2f} s · asimetría {fmt(sa,1)} % · mayor tiempo estimado: **{side}**. ({reference_text_for_metric('stance_asymmetry_2d')})")
            else:
                st.warning(f"Tiempo de apoyo NO informado: la segmentación apoyo/oscilación no superó el control temporal en ambos lados (I {fmt(qsl,0)}/100 · D {fmt(qsr,0)}/100). Se evita mostrar una duración potencialmente falsa.")
            st.markdown("#### Consistencia número de pasos · cadencia · duración")
            t1,t2,t3,t4 = st.columns(4)
            t1.metric("Eventos detectados", fmt(metric_value(metrics,"step_events_detected"),0))
            t2.metric("Duración segmento", fmt(metric_value(metrics,"segment_duration_s"),2)+" s")
            t3.metric("Eventos esperados", fmt(metric_value(metrics,"expected_steps_from_cadence"),1))
            t4.metric("Discrepancia", fmt(metric_value(metrics,"step_count_consistency_error_pct"),1)+" %")
            cq = metric_quality(metrics,"step_count_consistency_error_pct") or "No calculable"
            if cq == "Alta": st.success(f"Consistencia interna: {cq}")
            elif cq == "Moderada": st.info(f"Consistencia interna: {cq}")
            else: st.warning(f"Consistencia interna: {cq}. Conviene revisar visualmente los eventos detectados.")
            st.caption("Los eventos detectados son alternancias distales experimentales. La comprobación sirve para verificar coherencia interna, no para convertirlos automáticamente en heel-strikes clínicamente validados.")
            st.markdown("### Resumen biomecánico")
            for paragraph in biomech_summary(metrics, st.session_state.get("view", "")):
                st.write(paragraph)
            if "Frontal" in st.session_state.get("view", ""):
                st.markdown("### Biomecánica frontal/posterior 2D proyectada")
                st.warning("La rotación axial de cadera y la pronación son movimientos 3D. Aquí se muestran proxies 2D y no deben interpretarse como diagnóstico aislado.")
                for title, cols in [
                    ("Pelvis y tronco", ["Oblicuidad pélvica","Inclinación lateral del tronco","Relación hombros-pelvis"]),
                    ("Hombros", ["Oblicuidad de hombros"]),
                    ("Cadera / rodilla · eje frontal", ["Rodilla frontal izquierda","Rodilla frontal derecha"]),
                    ("Pie · orientación distal", ["Orientación pie izquierda","Orientación pie derecha"]),
                    ("Pie · inclinación del retropié", ["Retropié izquierda","Retropié derecha"]),
                    ("Avanzado frontal", ["CoM proxy lateral (px)","Valgo dinámico proyectado I","Valgo dinámico proyectado D"]),
                ]:
                    with st.expander(title, expanded=(title == "Pelvis y tronco")):
                        st.line_chart(chart.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo proyectado (°)")
            else:
                st.markdown("### Cinemática sagital 2D proyectada")
                for title, cols in [
                    ("Rodillas", ["Rodilla izquierda","Rodilla derecha"]),
                    ("Caderas", ["Cadera izquierda","Cadera derecha"]),
                    ("Tobillos / pie", ["Tobillo izquierda","Tobillo derecha"]),
                    ("Hombros", ["Hombro izquierda","Hombro derecha"]),
                ]:
                    with st.expander(title, expanded=(title == "Rodillas")):
                        st.line_chart(chart.set_index("time_s")[cols], x_label="Tiempo (s)", y_label="Ángulo 2D (°)")
            if st.session_state.get("annotated_video_bytes"):
                st.markdown("### Vídeo con esqueleto + ángulos")
                st.video(st.session_state.annotated_video_bytes)
        with st.expander("Todas las métricas y calidad"):
            _mdf=pd.DataFrame(metrics).copy()
            _mdf["reference"]=_mdf["key"].map(reference_text_for_metric)
            st.dataframe(_mdf[["label","value","unit","quality","reference","notes"]], use_container_width=True, hide_index=True)
        st.caption("Cadencia, alternancia y sincronización automática permanecen experimentales hasta validación específica del protocolo.")

with tabs[4]:
    st.subheader("Pacientes / Evolución longitudinal")
    st.caption("Paciente → registros → comparación → evolución. Los vídeos no se almacenan; el histórico usa sesiones y métricas guardadas en Supabase.")
    if not sb_ready():
        st.info("Configura Supabase para activar el histórico persistente.")
    else:
        try:
            pats = sb_list_patients()
            if pats.empty:
                st.info("Todavía no hay pacientes en el histórico.")
            else:
                codes = pats.code.tolist()
                default = codes.index(patient) if patient in codes else 0
                selected = st.selectbox("Paciente / código", codes, index=default, key="history_patient")
                hist_all = sb_patient_history(selected)
                if hist_all.empty:
                    st.info("Este paciente todavía no tiene métricas guardadas.")
                else:
                    hist_all["created_dt"] = pd.to_datetime(hist_all["created_at"], utc=True)
                    hist_all["fecha"] = hist_all["created_dt"].dt.strftime("%d/%m/%Y %H:%M")
                    sessions = hist_all[[c for c in ["session_id","created_dt","fecha","record_name","mode","view","assistive_device","frontal_orientation","duration_s","duration_cam2_s","segment_start_s","segment_end_s","analysis_status","ready_3d"] if c in hist_all.columns]].drop_duplicates("session_id").sort_values("created_dt")

                    a,b,c,d = st.columns(4)
                    a.metric("Registros", sessions.session_id.nunique())
                    a0=sessions.created_dt.min(); a1=sessions.created_dt.max()
                    b.metric("Primer registro", a0.strftime("%d/%m/%Y") if pd.notna(a0) else "—")
                    c.metric("Último registro", a1.strftime("%d/%m/%Y") if pd.notna(a1) else "—")
                    d.metric("Seguimiento", f"{max(0,(a1-a0).days)} días" if pd.notna(a0) and pd.notna(a1) else "—")

                    st.markdown("### Línea de tiempo de registros")
                    timeline=sessions.copy()
                    timeline["Registro"] = timeline["record_name"].fillna("Marcha")
                    timeline["Fecha"] = timeline["fecha"]
                    timeline["Vista"] = timeline["view"].fillna("")
                    timeline["Ayuda"] = timeline["assistive_device"].fillna("Sin ayuda")
                    timeline["Estado"] = timeline["analysis_status"].fillna("")
                    st.dataframe(timeline[["Fecha","Registro","Vista","Ayuda","Estado"]], use_container_width=True, hide_index=True)

                    with st.expander("🗑️ Gestionar / borrar registros duplicados", expanded=False):
                        st.caption(
                            "Selecciona uno o varios registros para eliminarlos del histórico. "
                            "Se borrarán la sesión y sus métricas de Supabase. El paciente no se elimina."
                        )
                        delete_labels = {
                            r.session_id: (
                                f"{r.fecha} · {r.record_name or 'Marcha'} · "
                                f"{r.view or 'Vista no especificada'} · "
                                f"{r.assistive_device or 'Sin ayuda'}"
                            )
                            for _, r in sessions.sort_values("created_dt", ascending=False).iterrows()
                        }
                        delete_ids = st.multiselect(
                            "Registros a borrar",
                            options=list(delete_labels.keys()),
                            format_func=lambda x: delete_labels[x],
                            key=f"delete_sessions_{selected}",
                            placeholder="Selecciona los registros duplicados o no válidos",
                        )
                        if delete_ids:
                            st.warning(
                                f"Vas a borrar permanentemente {len(delete_ids)} registro(s) "
                                "y todas sus métricas asociadas. Esta acción no se puede deshacer."
                            )
                            confirm_delete = st.checkbox(
                                "Confirmo que quiero borrar permanentemente los registros seleccionados",
                                key=f"confirm_delete_{selected}",
                            )
                            if st.button(
                                f"Eliminar {len(delete_ids)} registro(s)",
                                type="primary",
                                disabled=not confirm_delete,
                                key=f"delete_button_{selected}",
                            ):
                                deleted = 0
                                errors = []
                                for sid in delete_ids:
                                    try:
                                        sb_delete_session(sid)
                                        deleted += 1
                                    except Exception as exc:
                                        errors.append(f"{delete_labels.get(sid, sid)}: {exc}")
                                if deleted:
                                    st.success(
                                        f"Se han eliminado {deleted} registro(s) y sus métricas asociadas."
                                    )
                                if errors:
                                    st.error("No se pudieron borrar todos los registros:\n\n" + "\n".join(errors))
                                if deleted and not errors:
                                    st.rerun()

                    st.markdown("### Evolución de todos los registros")
                    f1,f2,f3 = st.columns(3)
                    aid_options=["Todas"]+sorted([x for x in hist_all.assistive_device.dropna().unique().tolist()]) if "assistive_device" in hist_all.columns else ["Todas"]
                    aid_filter=f1.selectbox("Ayuda técnica", aid_options, key="hist_aid")
                    view_options=["Todas"]+sorted([x for x in hist_all.view.dropna().unique().tolist()]) if "view" in hist_all.columns else ["Todas"]
                    view_filter=f2.selectbox("Vista", view_options, key="hist_view")
                    ref_mode=f3.selectbox("Referencia", ["Poblacional publicada", "Primer registro del paciente", "Sin referencia"], key="hist_ref")
                    hist=hist_all.copy()
                    if aid_filter!="Todas": hist=hist[hist.assistive_device==aid_filter]
                    if view_filter!="Todas": hist=hist[hist.view==view_filter]
                    labels = hist[["metric_key","metric_label","unit"]].drop_duplicates().sort_values("metric_label")
                    if labels.empty:
                        st.info("No hay métricas con esos filtros.")
                    else:
                        label = st.selectbox("Parámetro", labels.metric_label.tolist(), key="hist_metric")
                        key = labels.loc[labels.metric_label==label,"metric_key"].iloc[0]
                        h = hist[hist.metric_key==key].copy().sort_values("created_dt")
                        unit = h.unit.iloc[0] if len(h) else ""
                        if not h.empty:
                            plot=h[["fecha","value"]].dropna().set_index("fecha").rename(columns={"value":label})
                            st.line_chart(plot, x_label="Registro", y_label=f"{label} ({unit})" if unit else label)
                            first=float(h.value.dropna().iloc[0]) if h.value.notna().any() else np.nan
                            last=float(h.value.dropna().iloc[-1]) if h.value.notna().any() else np.nan
                            prev=float(h.value.dropna().iloc[-2]) if h.value.notna().sum()>=2 else np.nan
                            q1,q2,q3,q4=st.columns(4)
                            q1.metric("Basal", f"{first:.2f} {unit}" if np.isfinite(first) else "—")
                            q2.metric("Último", f"{last:.2f} {unit}" if np.isfinite(last) else "—", delta=f"{last-first:+.2f}" if np.isfinite(first) and np.isfinite(last) else None)
                            q3.metric("Δ vs basal", f"{((last-first)/abs(first)*100):+.1f} %" if np.isfinite(first) and first!=0 and np.isfinite(last) else "—")
                            q4.metric("Δ vs anterior", f"{last-prev:+.2f} {unit}" if np.isfinite(prev) and np.isfinite(last) else "—")
                            if ref_mode=="Poblacional publicada":
                                ref=reference_for_metric(key)
                                if ref:
                                    st.info(f"**Referencia contextual:** {ref['low']:.2f}–{ref['high']:.2f} {ref['unit']} · {reference_position(last,ref)}. {ref['population']}. {ref['applicability']}")
                                    st.caption(f"Fuente: {ref['source']} · DOI {ref['doi']} · Biblioteca PhysioSentinel {REFERENCE_LIBRARY_VERSION}")
                                else:
                                    st.warning("Esta métrica no tiene todavía una referencia poblacional suficientemente compatible con el método de PhysioSentinel. Se prioriza la evolución intraindividual.")
                            elif ref_mode=="Primer registro del paciente":
                                st.info("El primer registro filtrado se utiliza como referencia individual. Esto describe cambio respecto al basal y no implica por sí mismo mejoría o empeoramiento.")

                    st.markdown("### Comparar dos registros")
                    sess_labels={r.session_id:f"{r.fecha} · {r.record_name or 'Marcha'} · {r.view or ''} · {r.assistive_device or 'Sin ayuda'}" for _,r in sessions.iterrows()}
                    ids=list(sess_labels.keys())
                    if len(ids)<2:
                        st.info("Se necesitan al menos dos registros para una comparación directa.")
                    else:
                        ca,cb=st.columns(2)
                        sid_a=ca.selectbox("Registro A",ids,index=0,format_func=lambda x:sess_labels[x],key="cmp_a")
                        sid_b=cb.selectbox("Registro B",ids,index=len(ids)-1,format_func=lambda x:sess_labels[x],key="cmp_b")
                        if sid_a==sid_b:
                            st.warning("Selecciona dos registros diferentes.")
                        else:
                            ma=hist_all[hist_all.session_id==sid_a][["metric_key","metric_label","value","unit","quality"]].copy()
                            mb=hist_all[hist_all.session_id==sid_b][["metric_key","metric_label","value","unit","quality"]].copy()
                            cmp=ma.merge(mb,on="metric_key",suffixes=("_A","_B"))
                            cmp["Métrica"]=cmp.metric_label_A
                            cmp["A"]=cmp.value_A
                            cmp["B"]=cmp.value_B
                            cmp["Δ"]=cmp["B"]-cmp["A"]
                            cmp["Δ %"]=np.where(cmp["A"].abs()>1e-12,cmp["Δ"]/cmp["A"].abs()*100,np.nan)
                            cmp["Unidad"]=cmp.unit_A
                            st.dataframe(cmp[["Métrica","A","B","Δ","Δ %","Unidad"]].sort_values("Métrica"),use_container_width=True,hide_index=True)
                            sa=sessions[sessions.session_id==sid_a].iloc[0]; sb=sessions[sessions.session_id==sid_b].iloc[0]
                            comparable=(str(sa.get("view",""))==str(sb.get("view","")) and str(sa.get("assistive_device",""))==str(sb.get("assistive_device","")))
                            if comparable: st.success("Condiciones básicas comparables: misma vista y misma ayuda técnica.")
                            else: st.warning("Comparación contextual: cambia la vista y/o la ayuda técnica entre registros. Interpreta los Δ con cautela.")

                    st.markdown("### Resumen longitudinal")
                    st.write(f"**{selected}** dispone de **{sessions.session_id.nunique()} registros** entre {a0.strftime('%d/%m/%Y')} y {a1.strftime('%d/%m/%Y')}. La interpretación prioriza cambios intraindividuales y separa el valor medido de su significado clínico.")
                    st.caption("Un aumento o descenso no se etiqueta automáticamente como mejoría/empeoramiento. La dirección favorable depende de la variable, objetivo terapéutico, velocidad, ayuda técnica y contexto clínico.")
                    with st.expander("Biblioteca de referencias científicas"):
                        st.dataframe(pd.DataFrame(REFERENCE_SOURCES),use_container_width=True,hide_index=True)
                        st.caption("Las bandas solo se muestran cuando existe compatibilidad razonable con la métrica. No se trasladan rangos 3D a proxies 2D.")
        except Exception as e:
            st.error(f"No se pudo leer el histórico: {e}")

with tabs[5]:
    st.subheader("Redacción informe")
    st.caption("Informe clínico estructurado en 3 bloques + síntesis: espaciotemporales, cinemática frontal 2D, coordinación tronco-pelvis e impresión biomecánica. Cada cifra incluye referencia o precaución metodológica.")
    if not st.session_state.get("metrics_done"):
        st.info("Calcula primero el análisis en la pestaña 3 y los resultados en la pestaña 4.")
    else:
        metrics=st.session_state.get("metrics",[])
        tech,patient_txt=generate_reports(metrics, st.session_state.get("view",view), patient, record, assistive_device)
        st.markdown("### Informe de análisis biomecánico de la marcha (2D)")
        tech_edit=st.text_area("Texto técnico editable", value=tech, height=360, key="report_technical")
        st.markdown("### Versión simplificada para el paciente (complementaria)")
        pat_edit=st.text_area("Texto para paciente editable", value=patient_txt, height=320, key="report_patient")
        st.info("Convenio frontal documentado para pelvis: positivo (+) = lado izquierdo elevado / lado derecho descendido; negativo (-) = lado izquierdo descendido / lado derecho elevado. Para otras variables direccionales se evita atribuir signo anatómico cuando la orientación de cámara no permite hacerlo con seguridad.")
        full=f"PHYSIOSENTINEL GAIT {APP_VERSION}\n\nRESUMEN TÉCNICO-CLÍNICO\n{tech_edit}\n\nINFORME PARA EL PACIENTE\n{pat_edit}\n"
        st.download_button("Descargar informe TXT", data=full.encode("utf-8"), file_name=f"PhysioSentinel_Gait_{patient}_{record}.txt", mime="text/plain")

with tabs[6]:
    st.subheader("3D / Calibración")
    st.write("La v0.7 prepara el flujo para **triangulación 3D Pose2Sim** sin presentar todavía resultados 3D como si estuvieran calculados.")
    st.markdown("### 1. Perfil de calibración")
    st.caption("Carga aquí un archivo TOML de calibración de las dos cámaras generado/validado para tu montaje. Se guarda como texto pequeño en Supabase; los vídeos siguen sin persistir.")
    cal_name = st.text_input("Nombre del perfil", value="Consulta_2cam")
    cal_notes = st.text_input("Notas", value="Frontal/posterior + lateral; cámaras fijas")
    cal_file = st.file_uploader("Archivo de calibración TOML", type=["toml"], key="calibration_toml")
    if cal_file is not None:
        try:
            content = cal_file.getvalue().decode("utf-8")
            toml.loads(content)
            st.success("El archivo tiene sintaxis TOML válida. Esto no sustituye la validación geométrica de la calibración.")
            if st.button("Guardar / actualizar perfil en Supabase", type="primary"):
                if not sb_ready():
                    st.error("Supabase no está configurado.")
                elif not cal_name.strip():
                    st.error("Escribe un nombre para el perfil.")
                else:
                    row = sb_upsert_calibration(cal_name.strip(), content, cal_notes, camera_count=2)
                    st.session_state.selected_calibration_name = cal_name.strip()
                    if st.session_state.get("cloud_session_id"):
                        sb_update_session_3d(st.session_state.cloud_session_id, calibration_profile_name=cal_name.strip())
                    st.success(f"Perfil **{cal_name.strip()}** guardado. Ya puede seleccionarse para sesiones de 2 cámaras.")
        except Exception as e:
            st.error(f"El archivo no es TOML válido: {e}")

    st.markdown("### 2. Perfiles disponibles")
    if sb_ready():
        try:
            rows = sb_list_calibrations()
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Todavía no hay perfiles de calibración guardados.")
        except Exception as e:
            st.warning(f"No se pudieron leer perfiles: {e}")

    st.markdown("### 3. Estado de preparación 3D de la sesión actual")
    q1 = st.session_state.get("camera1_quality")
    q2 = st.session_state.get("camera2_quality")
    sq = st.session_state.get("sync_quality", "No calculable")
    ready, reasons = readiness_3d(st.session_state.get("mode",""), q1, q2, sq, st.session_state.get("selected_calibration_name"))
    if ready:
        st.success("La sesión cumple los criterios de preparación definidos en v0.7 para pasar a una futura triangulación 3D.")
    else:
        st.warning("Preparación 3D incompleta.")
        for reason in reasons:
            st.write(f"• {reason}")
    st.info("Siguiente etapa futura: materializar el perfil de calibración en /tmp, aplicar sincronización, ejecutar triangulación Pose2Sim, filtrar puntos 3D y posteriormente calcular cinemática/OpenSim. Esta v0.7 todavía no ejecuta esa etapa.")

st.divider()
st.caption("PhysioSentinel Gait v0.9.2 · línea temporal anatómica + cierre físico · CV robusto por lado · multipersona · stance por ciclos continuos · control de consistencia · informe técnico/paciente · 2 cámaras · preparación 3D")
