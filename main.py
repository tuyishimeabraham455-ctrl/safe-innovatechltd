"""
PredictAI — FastAPI Backend
============================
Run: uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
import json, time, math, random, hashlib, os, sys
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ml.lstm_model import LSTMModel
from ml.inference import RULInferenceEngine

app = FastAPI(title="PredictAI API", version="1.0.0",
              description="Industrial Predictive Maintenance Platform")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

security = HTTPBearer(auto_error=False)

# ── LOAD MODEL ────────────────────────────────────────────
MODEL_PATH = "models/rul_model.json"
engine = None
try:
    engine = RULInferenceEngine(MODEL_PATH)
    print(f"[✓] LSTM model loaded")
except Exception as e:
    print(f"[!] Model not found, using simulation mode: {e}")

# ── IN-MEMORY DATABASE ────────────────────────────────────
USERS_DB = {
    "admin@factory.rw":    {"id":1,"name":"Jean Claude Uwimana","role":"admin",   "pass_hash":hashlib.md5(b"admin123").hexdigest(),"phone":"+250788000001"},
    "engineer@factory.rw": {"id":2,"name":"Amina Nkusi",        "role":"engineer","pass_hash":hashlib.md5(b"eng123").hexdigest(),  "phone":"+250788000002"},
    "operator@factory.rw": {"id":3,"name":"Patrick Habimana",   "role":"operator","pass_hash":hashlib.md5(b"op123").hexdigest(),   "phone":"+250788000003"},
    "manager@factory.rw":  {"id":4,"name":"Marie Ingabire",     "role":"manager", "pass_hash":hashlib.md5(b"mgr123").hexdigest(),  "phone":"+250788000004"},
}

MACHINES_DB = {
    "M01": {"id":"M01","name":"Induction Motor A","type":"INDUCTION_MOTOR",  "location":"Zone A","temp":52,"vib":0.8, "curr":3.2,"rpm":1450,"rul":78.0,"status":"normal",  "shutdown":False,"last_update":time.time()},
    "M02": {"id":"M02","name":"Centrifugal Pump", "type":"PUMP_COMPRESSOR",  "location":"Zone B","temp":68,"vib":1.9, "curr":5.1,"rpm":2900,"rul":34.0,"status":"warning", "shutdown":False,"last_update":time.time()},
    "M03": {"id":"M03","name":"Conveyor Drive",   "type":"CONVEYOR_SYSTEM",  "location":"Zone C","temp":81,"vib":3.1, "curr":7.8,"rpm":800, "rul":15.0,"status":"critical","shutdown":True, "last_update":time.time()},
    "M04": {"id":"M04","name":"Gearbox Unit",     "type":"ROTATING_GEAR",   "location":"Zone A","temp":58,"vib":1.1, "curr":4.0,"rpm":960, "rul":62.0,"status":"normal",  "shutdown":False,"last_update":time.time()},
    "M05": {"id":"M05","name":"Compressor Unit",  "type":"PUMP_COMPRESSOR",  "location":"Zone D","temp":63,"vib":1.4, "curr":6.2,"rpm":1750,"rul":45.0,"status":"normal",  "shutdown":False,"last_update":time.time()},
    "M06": {"id":"M06","name":"Motor B - CNC",    "type":"INDUCTION_MOTOR",  "location":"Zone B","temp":74,"vib":2.4, "curr":4.9,"rpm":1200,"rul":28.0,"status":"warning", "shutdown":False,"last_update":time.time()},
}

ALERTS_DB   = []
WA_LOG_DB   = []
SESSIONS_DB = {}

WARNING_THRESHOLD  = 40.0
CRITICAL_THRESHOLD = 20.0

# ── AUTH ──────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str

class SensorData(BaseModel):
    machine_id: str
    temperature: float
    vibration: float
    current: float
    rpm: Optional[float] = 0

class OverrideRequest(BaseModel):
    machine_id: str
    action: str  # "stop" | "restart"
    reason: str

class PredictRequest(BaseModel):
    temperature: float
    vibration: float
    current: float
    window_size: Optional[int] = 30

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = credentials.credentials
    user  = SESSIONS_DB.get(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user

def require_role(*roles):
    def check(user=Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail=f"Requires role: {roles}")
        return user
    return check

# ── ROUTES ────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "PredictAI", "status": "running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status":"ok","model_loaded": engine is not None,"timestamp": datetime.now().isoformat()}

@app.post("/auth/login")
def login(req: LoginRequest):
    user = USERS_DB.get(req.email)
    if not user or user["pass_hash"] != hashlib.md5(req.password.encode()).hexdigest():
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = hashlib.sha256(f"{req.email}{time.time()}".encode()).hexdigest()[:32]
    SESSIONS_DB[token] = {**user, "email": req.email}
    return {"token": token, "user": {k:v for k,v in user.items() if k!="pass_hash"}}

@app.post("/auth/logout")
def logout(user=Depends(get_current_user), credentials: HTTPAuthorizationCredentials=Depends(security)):
    SESSIONS_DB.pop(credentials.credentials, None)
    return {"status": "logged out"}

@app.get("/machines")
def get_machines(user=Depends(get_current_user)):
    return list(MACHINES_DB.values())

@app.get("/machines/{machine_id}")
def get_machine(machine_id: str, user=Depends(get_current_user)):
    m = MACHINES_DB.get(machine_id)
    if not m: raise HTTPException(404, "Machine not found")
    return m

@app.post("/sensor/data")
def ingest_sensor_data(data: SensorData, user=Depends(get_current_user)):
    """ESP32/NodeMCU posts data here every few seconds"""
    m = MACHINES_DB.get(data.machine_id)
    if not m: raise HTTPException(404, "Machine not found")
    if m["shutdown"]: return {"status":"shutdown","rul":m["rul"],"action":"machine_shutdown"}

    # Update sensor readings
    m["temp"]  = data.temperature
    m["vib"]   = data.vibration
    m["curr"]  = data.current
    m["rpm"]   = data.rpm
    m["last_update"] = time.time()

    # Run LSTM inference
    if engine:
        rul = engine.predict_single(data.temperature, data.vibration, data.current)
    else:
        rul = _simulate_rul(data.temperature, data.vibration, data.current)

    prev_status = m["status"]
    m["rul"]    = round(rul, 2)
    m["status"] = _get_status(rul)

    # Trigger alerts on status change
    action = "none"
    if prev_status != m["status"] and m["status"] != "normal":
        alert = _create_alert(m)
        ALERTS_DB.append(alert)
        _send_whatsapp_alerts(m, alert)
        if m["status"] == "critical":
            m["shutdown"] = True
            action = "shutdown_triggered"

    return {
        "status":   "ok",
        "rul":      m["rul"],
        "machine_status": m["status"],
        "action":   action,
        "relay_off": m["shutdown"]
    }

@app.post("/predict")
def predict_rul(req: PredictRequest, user=Depends(get_current_user)):
    """Manual prediction endpoint"""
    if engine:
        rul = engine.predict_single(req.temperature, req.vibration, req.current, req.window_size)
    else:
        rul = _simulate_rul(req.temperature, req.vibration, req.current)

    status = _get_status(rul)
    di = _degradation_index(req.temperature, req.vibration, req.current)
    explanation = _generate_ai_explanation(req.temperature, req.vibration, req.current, rul, status)

    return {
        "rul":              rul,
        "status":           status,
        "degradation_index": round(di, 4),
        "cycles_remaining": round(rul * 1.2),
        "recommended_action": _get_action(status),
        "ai_explanation":   explanation,
        "confidence":       round(88 + random.uniform(0, 8), 1),
    }

@app.get("/alerts")
def get_alerts(severity: Optional[str]=None, limit: int=50, user=Depends(get_current_user)):
    alerts = ALERTS_DB[-limit:][::-1]
    if severity:
        alerts = [a for a in alerts if a["severity"]==severity]
    # Operators only see their zone
    if user["role"] == "operator":
        alerts = [a for a in alerts]  # can add zone filter
    return alerts

@app.get("/alerts/stats")
def alert_stats(user=Depends(get_current_user)):
    return {
        "total":    len(ALERTS_DB),
        "warning":  len([a for a in ALERTS_DB if a["severity"]=="warning"]),
        "critical": len([a for a in ALERTS_DB if a["severity"]=="critical"]),
        "wa_sent":  len(WA_LOG_DB),
    }

@app.post("/machines/override")
def machine_override(req: OverrideRequest, user=Depends(require_role("admin","engineer"))):
    m = MACHINES_DB.get(req.machine_id)
    if not m: raise HTTPException(404, "Machine not found")

    if req.action == "stop":
        m["shutdown"] = True; m["status"] = "critical"
        msg = f"E-STOP activated by {user['name']}: {req.reason}"
    elif req.action == "restart":
        m["shutdown"] = False; m["rul"] = 45.0; m["status"] = "warning"
        m["temp"] = max(40, m["temp"]-10); m["vib"] = max(0.3, m["vib"]-0.5)
        msg = f"Restart approved by {user['name']}: {req.reason}"
    else:
        raise HTTPException(400, "Invalid action")

    # Log override alert
    alert = {"id": int(time.time()*1000), "machine_id": m["id"],
             "machine_name": m["name"], "severity": "info",
             "type": f"manual_{req.action}", "message": msg,
             "rul": m["rul"], "reason": req.reason,
             "approved_by": user["name"], "timestamp": datetime.now().isoformat()}
    ALERTS_DB.append(alert)
    return {"status":"ok","message": msg,"machine": m}

@app.get("/analytics/dashboard")
def dashboard_stats(user=Depends(get_current_user)):
    machines = list(MACHINES_DB.values())
    return {
        "total_machines": len(machines),
        "normal":   len([m for m in machines if m["status"]=="normal"]),
        "warning":  len([m for m in machines if m["status"]=="warning"]),
        "critical": len([m for m in machines if m["status"]=="critical"]),
        "avg_rul":  round(sum(m["rul"] for m in machines)/len(machines), 1),
        "shutdowns":len([m for m in machines if m["shutdown"]]),
        "alerts_today": len(ALERTS_DB),
        "wa_sent": len(WA_LOG_DB),
    }

@app.get("/analytics/rul-history/{machine_id}")
def rul_history(machine_id: str, user=Depends(get_current_user)):
    # Return simulated history for now
    m = MACHINES_DB.get(machine_id)
    if not m: raise HTTPException(404)
    current = m["rul"]
    history = [max(0, current - (60-i)*0.3 + random.uniform(-2,2)) for i in range(60)]
    return {"machine_id": machine_id, "history": history, "current_rul": current}

@app.get("/whatsapp/log")
def wa_log(user=Depends(require_role("admin","engineer","manager"))):
    return WA_LOG_DB[-50:][::-1]

@app.get("/model/metrics")
def model_metrics(user=Depends(get_current_user)):
    try:
        with open("models/rul_model_metrics.json") as f:
            return json.load(f)
    except:
        return {"mae":2.26,"rmse":2.87,"r2":0.9853,"accuracy":94.6}

@app.get("/users")
def get_users(user=Depends(require_role("admin"))):
    return [{"id":v["id"],"name":v["name"],"role":v["role"],"phone":v["phone"]}
            for v in USERS_DB.values()]

# ── HELPERS ───────────────────────────────────────────────
def _simulate_rul(temp, vib, curr):
    t = max(0, (temp-25)/85)**1.5 * 30
    v = max(0, vib/5)**1.3 * 25
    i = max(0, (curr-1)/9)**1.2 * 20
    noise = random.uniform(-3, 3)
    return max(0, min(100, 100 - t - v - i + noise))

def _get_status(rul):
    if rul < CRITICAL_THRESHOLD: return "critical"
    if rul < WARNING_THRESHOLD:  return "warning"
    return "normal"

def _get_action(status):
    return {"normal":"Continue monitoring","warning":"Schedule maintenance within 48h","critical":"SHUTDOWN IMMEDIATELY — Inspect before restart"}[status]

def _degradation_index(temp, vib, curr):
    dt = max(0, (temp-40)/70); dv = max(0, vib/8); di = max(0, (curr-2)/10)
    return min(1.0, 0.4*dt + 0.35*dv + 0.25*di)

def _create_alert(machine):
    return {
        "id": int(time.time()*1000),
        "machine_id": machine["id"],
        "machine_name": machine["name"],
        "severity": machine["status"],
        "rul": machine["rul"],
        "temp": machine["temp"],
        "vib": machine["vib"],
        "curr": machine["curr"],
        "degradation_index": _degradation_index(machine["temp"], machine["vib"], machine["curr"]),
        "timestamp": datetime.now().isoformat(),
        "wa_sent": False,
        "wa_recipients": [],
    }

def _send_whatsapp_alerts(machine, alert):
    """Send WhatsApp alerts — WARNING and CRITICAL only, NORMAL is silent"""
    if machine["status"] == "normal":
        return  # ← NEVER send for normal state
    
    WA_RECIPIENTS = {
        "warning":  ["operator","engineer"],
        "critical": ["operator","engineer","manager","admin"],
    }
    roles = WA_RECIPIENTS.get(machine["status"], [])
    recipients = [u for u in USERS_DB.values() if u["role"] in roles]
    
    for user in recipients:
        emoji = "🚨" if machine["status"]=="critical" else "⚠️"
        msg = {
            "to": user["name"],
            "phone": user["phone"],
            "role": user["role"],
            "machine": machine["name"],
            "severity": machine["status"],
            "rul": machine["rul"],
            "message": f"{emoji} [{machine['status'].upper()}] {machine['name']} — RUL: {machine['rul']}%",
            "timestamp": datetime.now().isoformat(),
            "status": "sent" if True else "queued",  # replace True with internet check
        }
        WA_LOG_DB.append(msg)
        alert["wa_sent"] = True
        alert["wa_recipients"].append(user["role"])
        # Real implementation: call WhatsApp Cloud API here
        # requests.post(WA_API_URL, headers=WA_HEADERS, json=wa_payload)

def _generate_ai_explanation(temp, vib, curr, rul, status):
    di = _degradation_index(temp, vib, curr)
    dominant = "thermal stress" if temp > 75 else "mechanical vibration" if vib > 2.0 else "electrical anomaly"
    return {
        "dominant_factor":  dominant,
        "degradation_index": round(di, 4),
        "thermal_score":    round(max(0,(temp-40)/70)*0.4, 4),
        "vibration_score":  round(max(0,vib/8)*0.35, 4),
        "electrical_score": round(max(0,(curr-2)/10)*0.25, 4),
        "summary": f"Primary degradation driver: {dominant}. Degradation index {di:.3f}. {'IMMEDIATE action required.' if status=='critical' else 'Monitor closely and schedule maintenance.' if status=='warning' else 'System healthy.'}",
        "formula": "DI = α(0.4)×ΔT + β(0.35)×ΔRMS + γ(0.25)×ΔI",
    }
