"""
PredictAI — WhatsApp Cloud API Integration
==========================================
Sends WARNING and CRITICAL alerts to roles.
NORMAL state = SILENT (no messages sent).

Setup:
  1. Go to https://developers.facebook.com
  2. Create App → Add WhatsApp product
  3. Get Phone Number ID + Access Token
  4. Set env vars or edit config below
"""

import os
import json
import time
import queue
import threading
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────
#  CONFIG — set via environment or edit directly
# ─────────────────────────────────────────────────────
WA_CONFIG = {
    "access_token":    os.getenv("WA_ACCESS_TOKEN",    "YOUR_ACCESS_TOKEN"),
    "phone_number_id": os.getenv("WA_PHONE_NUMBER_ID", "YOUR_PHONE_NUMBER_ID"),
    "api_version":     "v18.0",
    "api_url":         "https://graph.facebook.com",
}

# Role → phone numbers (edit with real numbers)
RECIPIENT_PHONES = {
    "operator": os.getenv("WA_OPERATOR_PHONE", "+250788000003"),
    "engineer": os.getenv("WA_ENGINEER_PHONE", "+250788000002"),
    "manager":  os.getenv("WA_MANAGER_PHONE",  "+250788000004"),
    "admin":    os.getenv("WA_ADMIN_PHONE",    "+250788000001"),
}

# Who gets what severity — NORMAL IS NEVER SENT
ALERT_RECIPIENTS = {
    "warning":  ["operator", "engineer"],
    "critical": ["operator", "engineer", "manager"],
    "shutdown": ["operator", "engineer", "manager", "admin"],
}

# ─────────────────────────────────────────────────────
#  MESSAGE BUILDER
# ─────────────────────────────────────────────────────
def build_message(machine: dict, severity: str, recipient_role: str, is_shutdown: bool = False) -> str:
    """Build WhatsApp message text for given machine and severity."""
    # NEVER build for normal
    if severity == "normal":
        return None

    emoji_map = {"warning": "⚠️", "critical": "🚨"}
    emoji = emoji_map.get(severity, "📢")

    if is_shutdown:
        action_line = "🔴 AUTO-SHUTDOWN ACTIVATED — Relay cut power.\n   ⚙️ Engineer inspection + approval required before restart."
    elif severity == "critical":
        action_line = "🔴 IMMEDIATE ACTION REQUIRED\n   Do NOT restart without full inspection."
    else:
        action_line = "🔧 Schedule maintenance within 48 hours.\n   Increase monitoring frequency."

    role_suffix = {
        "operator": "\n\nOperator: Please do not restart the machine.",
        "engineer": "\n\nEngineer: Inspection and restart approval required.",
        "manager":  "\n\nManager: Production impact assessment needed.",
        "admin":    "\n\nAdmin: System logged. Review override policy.",
    }.get(recipient_role, "")

    di = machine.get("degradation_index", 0)
    msg = f"""{emoji} PREDICTAI ALERT — {severity.upper()}

Machine : {machine.get('name', 'Unknown')} ({machine.get('id', '?')})
Location: {machine.get('location', 'Unknown')}
Status  : {severity.upper()}

📊 Live Sensor Data:
  • RUL      : {machine.get('rul', 0):.1f}% remaining
  • Temp     : {machine.get('temp', 0):.1f}°C
  • Vibration: {machine.get('vib', 0):.2f}g RMS
  • Current  : {machine.get('curr', 0):.1f}A
  • Degrad.  : {di:.4f}

🤖 AI Recommendation:
  {action_line}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{role_suffix}

— PredictAI Maintenance Platform"""
    return msg


# ─────────────────────────────────────────────────────
#  WHATSAPP SENDER
# ─────────────────────────────────────────────────────
class WhatsAppAlerter:
    """
    Sends WhatsApp messages via Meta Cloud API.
    Includes offline queue with automatic retry.
    """

    def __init__(self):
        self._queue    = queue.Queue()
        self._sent     = []
        self._failed   = []
        self._online   = True
        self._lock     = threading.Lock()
        # Start background retry thread
        threading.Thread(target=self._retry_loop, daemon=True).start()

    def send_alert(self, machine: dict, severity: str, is_shutdown: bool = False) -> dict:
        """
        Main entry point. Only sends for WARNING or CRITICAL.
        NORMAL severity → returns immediately with no action.
        """
        if severity == "normal":
            return {"status": "skipped", "reason": "normal_state_silent"}

        alert_type = "shutdown" if is_shutdown else severity
        roles = ALERT_RECIPIENTS.get(alert_type, ALERT_RECIPIENTS.get(severity, []))

        results = []
        for role in roles:
            phone = RECIPIENT_PHONES.get(role)
            if not phone:
                continue
            message = build_message(machine, severity, role, is_shutdown)
            if not message:
                continue

            entry = {
                "id":        int(time.time() * 1000),
                "machine":   machine.get("name"),
                "severity":  severity,
                "role":      role,
                "phone":     phone,
                "message":   message,
                "timestamp": datetime.now().isoformat(),
                "status":    "pending",
                "attempts":  0,
            }

            if self._online:
                result = self._send_now(entry)
                results.append(result)
            else:
                # Queue for retry
                self._queue.put(entry)
                logger.warning(f"[WA] Offline — queued alert for {role}")
                results.append({"status": "queued", "role": role, "phone": phone})

        return {
            "status":     "sent" if all(r.get("status") == "sent" for r in results) else "partial",
            "recipients": results,
            "severity":   severity,
            "machine":    machine.get("name"),
        }

    def _send_now(self, entry: dict) -> dict:
        """Send via WhatsApp Cloud API."""
        try:
            # ── REAL API CALL ────────────────────────────────────
            # Uncomment this block when you have real credentials:
            #
            # import requests
            # url = f"{WA_CONFIG['api_url']}/{WA_CONFIG['api_version']}/{WA_CONFIG['phone_number_id']}/messages"
            # headers = {
            #     "Authorization": f"Bearer {WA_CONFIG['access_token']}",
            #     "Content-Type": "application/json",
            # }
            # payload = {
            #     "messaging_product": "whatsapp",
            #     "to": entry["phone"],
            #     "type": "text",
            #     "text": {"body": entry["message"]},
            # }
            # resp = requests.post(url, headers=headers, json=payload, timeout=10)
            # if resp.status_code != 200:
            #     raise Exception(f"API error {resp.status_code}: {resp.text}")
            # ── END REAL API CALL ────────────────────────────────

            # Simulation mode (remove when using real API)
            logger.info(f"[WA] SIMULATED → {entry['role']} ({entry['phone']}): {entry['severity'].upper()}")

            entry["status"]   = "sent"
            entry["sent_at"]  = datetime.now().isoformat()
            entry["attempts"] += 1

            with self._lock:
                self._sent.append(entry)

            return {"status": "sent", "role": entry["role"], "phone": entry["phone"]}

        except Exception as e:
            entry["status"]   = "failed"
            entry["error"]    = str(e)
            entry["attempts"] += 1
            logger.error(f"[WA] Send failed: {e}")
            with self._lock:
                self._failed.append(entry)
            return {"status": "failed", "role": entry["role"], "error": str(e)}

    def _retry_loop(self):
        """Background thread: retry queued messages when online."""
        while True:
            time.sleep(30)
            if not self._queue.empty():
                logger.info(f"[WA] Retrying {self._queue.qsize()} queued messages...")
                temp = []
                while not self._queue.empty():
                    temp.append(self._queue.get())
                for entry in temp:
                    result = self._send_now(entry)
                    if result["status"] != "sent":
                        self._queue.put(entry)  # re-queue

    def set_online(self, status: bool):
        self._online = status

    def get_log(self, limit: int = 50) -> list:
        with self._lock:
            return list(reversed(self._sent[-limit:]))

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "sent":    len(self._sent),
                "failed":  len(self._failed),
                "queued":  self._queue.qsize(),
                "online":  self._online,
            }


# Singleton
alerter = WhatsAppAlerter()


# ─────────────────────────────────────────────────────
#  CONVENIENCE FUNCTIONS
# ─────────────────────────────────────────────────────
def send_warning_alert(machine: dict):
    """Send WARNING alert — operators + engineers."""
    return alerter.send_alert(machine, "warning")

def send_critical_alert(machine: dict):
    """Send CRITICAL alert — all roles."""
    return alerter.send_alert(machine, "critical")

def send_shutdown_alert(machine: dict):
    """Send SHUTDOWN notification — all roles including admin."""
    return alerter.send_alert(machine, "critical", is_shutdown=True)

def silence_normal():
    """Normal state — never sends. Returns silenced status."""
    return {"status": "silenced", "reason": "normal_operation"}


# ─────────────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_machine = {
        "id": "M02", "name": "Centrifugal Pump",
        "location": "Zone B", "rul": 32.5,
        "temp": 71.0, "vib": 2.1, "curr": 5.8,
        "degradation_index": 0.6230,
    }

    print("\n── Test 1: Normal (should be SILENT) ──")
    r = alerter.send_alert(test_machine, "normal")
    print(f"Result: {r}")

    print("\n── Test 2: Warning Alert ──")
    r = alerter.send_alert(test_machine, "warning")
    print(f"Sent to: {[x['role'] for x in r.get('recipients', [])]}")

    print("\n── Test 3: Critical + Shutdown ──")
    r = alerter.send_alert(test_machine, "critical", is_shutdown=True)
    print(f"Sent to: {[x['role'] for x in r.get('recipients', [])]}")

    print(f"\n── Stats: {alerter.get_stats()}")
    print("\n✅ WhatsApp alerter test complete")
