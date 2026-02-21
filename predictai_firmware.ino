/*
 * PredictAI — ESP32 / NodeMCU Firmware
 * ======================================
 * Reads Temperature (DS18B20), Vibration (MPU6050), Current (ACS712)
 * Runs local RUL threshold check (offline-safe)
 * Posts data to PredictAI backend via WiFi
 * Controls relay for auto-shutdown
 * Stores to SD card when offline
 *
 * Board:    ESP32 (38-pin DevKit)
 * IDE:      Arduino IDE 2.x
 * Required Libraries:
 *   - OneWire           (Paul Stoffregen)
 *   - DallasTemperature (Miles Burton)
 *   - MPU6050           (Electronic Cats)
 *   - ArduinoJson       (Benoit Blanchon) v6+
 *   - WiFi              (built-in ESP32)
 *   - HTTPClient        (built-in ESP32)
 *   - SD                (built-in ESP32)
 */

#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>
#include "SD.h"
#include "SPI.h"
#include <math.h>

// ─────────────────────────────────────────────────────
//  PIN DEFINITIONS
// ─────────────────────────────────────────────────────
#define ONE_WIRE_BUS    4       // DS18B20 data pin
#define ACS712_PIN      34      // ACS712 analog (ADC1 only!)
#define RELAY_PIN       26      // Relay control
#define SD_CS_PIN       5       // SD card chip select
#define LED_NORMAL      2       // Green LED (onboard)
#define LED_WARNING     15      // Orange LED
#define LED_CRITICAL    13      // Red LED

// MPU6050 uses I2C: SDA=21, SCL=22 (ESP32 default)

// ─────────────────────────────────────────────────────
//  CONFIGURATION — EDIT THESE
// ─────────────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* API_URL       = "http://YOUR_SERVER_IP:8000";
const char* AUTH_TOKEN    = "YOUR_JWT_TOKEN";
const char* MACHINE_ID    = "M01";
const char* MACHINE_TYPE  = "INDUCTION_MOTOR";

// Thresholds (fallback if API unreachable)
const float RUL_WARNING_THRESHOLD  = 40.0;
const float RUL_CRITICAL_THRESHOLD = 20.0;
const float TEMP_MAX   = 85.0;   // °C
const float VIB_MAX    = 3.0;    // g RMS
const float CURR_MAX   = 10.0;   // A

const int   SAMPLE_RATE_MS  = 3000;   // 3 second intervals
const int   WINDOW_SIZE     = 30;     // matches training
const int   RETRY_INTERVAL  = 60000; // 60s offline retry

// ─────────────────────────────────────────────────────
//  SENSOR OBJECTS
// ─────────────────────────────────────────────────────
OneWire            oneWire(ONE_WIRE_BUS);
DallasTemperature  tempSensor(&oneWire);

// MPU6050 registers (manual I2C, no lib dependency)
#define MPU6050_ADDR     0x68
#define MPU6050_PWR_MGMT 0x6B
#define MPU6050_ACCEL_X  0x3B

// ACS712 config (ACS712-05B: 185mV/A, ACS712-20A: 100mV/A)
#define ACS712_SENSITIVITY  0.185  // V/A for 5A model
#define ACS712_VREF         2.5    // V midpoint
#define ACS712_ADC_MAX      4095
#define ACS712_VCC          3.3

// ─────────────────────────────────────────────────────
//  STATE
// ─────────────────────────────────────────────────────
struct SensorReading {
  float temperature;
  float vibration_rms;
  float current;
  unsigned long timestamp;
};

struct RULResult {
  float  rul;
  String status;     // "normal" | "warning" | "critical"
  float  degradation_index;
  bool   relay_off;
};

SensorReading history[WINDOW_SIZE];
int           historyIdx   = 0;
bool          historyFull  = false;
bool          relayTripped = false;
unsigned long lastPost     = 0;
unsigned long lastWiFiTry  = 0;
bool          wifiConnected = false;
float         lastRUL      = 100.0;

// Offline queue (SD card)
bool sdAvailable = false;

// ─────────────────────────────────────────────────────
//  SETUP
// ─────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.println("\n[PredictAI] ESP32 Firmware v1.0");
  Serial.println("[PredictAI] Booting...");

  // GPIO
  pinMode(RELAY_PIN,    OUTPUT);
  pinMode(LED_NORMAL,   OUTPUT);
  pinMode(LED_WARNING,  OUTPUT);
  pinMode(LED_CRITICAL, OUTPUT);
  digitalWrite(RELAY_PIN, HIGH);  // Relay NC = machine ON when HIGH

  // Boot blink
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_NORMAL, HIGH); delay(200);
    digitalWrite(LED_NORMAL, LOW);  delay(200);
  }

  // Sensors
  tempSensor.begin();
  Serial.printf("[OK] DS18B20: %d sensor(s) found\n", tempSensor.getDeviceCount());

  mpu6050_init();
  Serial.println("[OK] MPU6050 initialized");

  // SD card
  sdAvailable = SD.begin(SD_CS_PIN);
  Serial.printf("[%s] SD Card\n", sdAvailable ? "OK" : "!!") ;

  // WiFi
  connectWiFi();

  Serial.println("[PredictAI] Ready. Starting sensor loop...\n");
}

// ─────────────────────────────────────────────────────
//  MAIN LOOP
// ─────────────────────────────────────────────────────
void loop() {
  if (relayTripped) {
    // Machine is shutdown — blink critical LED, wait for restart
    digitalWrite(LED_CRITICAL, HIGH); delay(500);
    digitalWrite(LED_CRITICAL, LOW);  delay(500);
    return;
  }

  unsigned long now = millis();
  if (now - lastPost < SAMPLE_RATE_MS) return;
  lastPost = now;

  // 1. Read sensors
  SensorReading reading = readSensors();
  printReading(reading);

  // 2. Store in history window
  history[historyIdx] = reading;
  historyIdx = (historyIdx + 1) % WINDOW_SIZE;
  if (historyIdx == 0) historyFull = true;

  // 3. Local RUL estimation (offline-safe)
  RULResult local = localRULEstimate(reading);
  Serial.printf("[LSTM] Local RUL: %.1f%% | Status: %s\n",
                local.rul, local.status.c_str());

  // 4. Update relay based on local estimate
  handleSafety(local);

  // 5. Post to API (if WiFi available)
  if (wifiConnected) {
    postToAPI(reading, local);
  } else {
    saveToSD(reading, local);
    // Try reconnect every 60s
    if (now - lastWiFiTry > RETRY_INTERVAL) {
      connectWiFi();
      lastWiFiTry = now;
    }
  }

  // 6. Update LEDs
  updateLEDs(local.status);
}

// ─────────────────────────────────────────────────────
//  SENSOR READING
// ─────────────────────────────────────────────────────
SensorReading readSensors() {
  SensorReading r;
  r.timestamp = millis();

  // ── Temperature ──
  tempSensor.requestTemperatures();
  float rawTemp = tempSensor.getTempCByIndex(0);
  r.temperature = (rawTemp == DEVICE_DISCONNECTED_C) ? 0.0 : rawTemp;

  // ── Vibration (MPU6050 RMS) ──
  float ax, ay, az;
  mpu6050_readAccel(ax, ay, az);
  r.vibration_rms = sqrt(ax*ax + ay*ay + az*az);  // g RMS magnitude

  // ── Current (ACS712 - 50 samples average) ──
  float currentSum = 0;
  for (int s = 0; s < 50; s++) {
    int raw = analogRead(ACS712_PIN);
    float voltage = (raw / (float)ACS712_ADC_MAX) * ACS712_VCC;
    currentSum += abs((voltage - ACS712_VREF) / ACS712_SENSITIVITY);
    delayMicroseconds(100);
  }
  r.current = currentSum / 50.0;

  return r;
}

// ─────────────────────────────────────────────────────
//  LOCAL LSTM RUL ESTIMATION (offline-safe)
//  Simplified version of the Python LSTM model
//  Uses degradation formula: DI = α×ΔT + β×ΔRMS + γ×ΔI
// ─────────────────────────────────────────────────────
RULResult localRULEstimate(SensorReading& r) {
  // Window averages
  float avgTemp = 0, avgVib = 0, avgCurr = 0;
  int   count   = historyFull ? WINDOW_SIZE : historyIdx;
  if (count == 0) count = 1;

  for (int i = 0; i < count; i++) {
    avgTemp += history[i].temperature;
    avgVib  += history[i].vibration_rms;
    avgCurr += history[i].current;
  }
  avgTemp /= count; avgVib /= count; avgCurr /= count;

  // Variance
  float varT=0, varV=0, varI=0;
  for (int i = 0; i < count; i++) {
    varT += sq(history[i].temperature   - avgTemp);
    varV += sq(history[i].vibration_rms - avgVib);
    varI += sq(history[i].current       - avgCurr);
  }
  varT /= count; varV /= count; varI /= count;

  // Degradation index (matches Python model formula)
  float alpha=0.4, beta=0.35, gamma=0.25;
  float dT = constrain((avgTemp - 40.0) / 70.0, 0, 1);
  float dV = constrain(avgVib / 8.0,             0, 1);
  float dI = constrain((avgCurr - 2.0) / 10.0,  0, 1);
  float DI = alpha*dT + beta*dV + gamma*dI + 0.02*(varT+varV+varI);
  DI = constrain(DI, 0.0, 1.0);

  // RUL estimate
  float rul = 100.0 * (1.0 - sq(DI));
  rul = constrain(rul, 0.0, 100.0);

  // Smooth
  rul = 0.7 * rul + 0.3 * lastRUL;
  lastRUL = rul;

  RULResult res;
  res.rul = rul;
  res.degradation_index = DI;
  res.relay_off = false;

  if (rul < RUL_CRITICAL_THRESHOLD || r.temperature > TEMP_MAX ||
      r.vibration_rms > VIB_MAX    || r.current > CURR_MAX) {
    res.status    = "critical";
    res.relay_off = true;
  } else if (rul < RUL_WARNING_THRESHOLD) {
    res.status = "warning";
  } else {
    res.status = "normal";
  }

  return res;
}

// ─────────────────────────────────────────────────────
//  SAFETY — RELAY CONTROL
// ─────────────────────────────────────────────────────
void handleSafety(RULResult& res) {
  if (res.relay_off && !relayTripped) {
    // SHUTDOWN
    digitalWrite(RELAY_PIN, LOW);  // Cut power to machine
    relayTripped = true;
    Serial.println("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    Serial.println("!! AUTO-SHUTDOWN TRIGGERED             !!");
    Serial.printf ("!! RUL: %.1f%% — RELAY OFF              !!\n", res.rul);
    Serial.println("!! Engineer restart approval required  !!");
    Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  }
}

// ─────────────────────────────────────────────────────
//  API POST
// ─────────────────────────────────────────────────────
void postToAPI(SensorReading& r, RULResult& local) {
  HTTPClient http;
  String url = String(API_URL) + "/sensor/data";
  http.begin(url);
  http.addHeader("Content-Type",  "application/json");
  http.addHeader("Authorization", String("Bearer ") + AUTH_TOKEN);
  http.setTimeout(5000);

  StaticJsonDocument<256> doc;
  doc["machine_id"]  = MACHINE_ID;
  doc["temperature"] = r.temperature;
  doc["vibration"]   = r.vibration_rms;
  doc["current"]     = r.current;
  doc["rpm"]         = 1450;  // add RPM sensor if available

  String payload;
  serializeJson(doc, payload);

  int code = http.POST(payload);

  if (code == 200) {
    String resp = http.getString();
    StaticJsonDocument<256> respDoc;
    deserializeJson(respDoc, resp);

    float  apiRUL    = respDoc["rul"]    | local.rul;
    String apiStatus = respDoc["machine_status"] | local.status;
    bool   shutdown  = respDoc["relay_off"] | false;

    Serial.printf("[API ] RUL: %.1f%% | Status: %s\n", apiRUL, apiStatus.c_str());

    if (shutdown && !relayTripped) {
      RULResult fake; fake.relay_off = true; fake.rul = apiRUL;
      handleSafety(fake);
    }
  } else {
    Serial.printf("[WARN] API error %d — using local estimate\n", code);
    wifiConnected = false;
  }
  http.end();
}

// ─────────────────────────────────────────────────────
//  SD CARD OFFLINE LOGGING
// ─────────────────────────────────────────────────────
void saveToSD(SensorReading& r, RULResult& res) {
  if (!sdAvailable) return;
  File f = SD.open("/data_log.csv", FILE_APPEND);
  if (!f) return;
  f.printf("%lu,%.2f,%.3f,%.2f,%.1f,%s\n",
           r.timestamp, r.temperature, r.vibration_rms,
           r.current,   res.rul, res.status.c_str());
  f.close();
  Serial.println("[SD  ] Reading saved offline");
}

// ─────────────────────────────────────────────────────
//  WIFI
// ─────────────────────────────────────────────────────
void connectWiFi() {
  Serial.printf("[WiFi] Connecting to %s", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 20) {
    delay(500); Serial.print(".");
    tries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    Serial.printf("\n[WiFi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    wifiConnected = false;
    Serial.println("\n[WiFi] Failed — OFFLINE MODE");
  }
}

// ─────────────────────────────────────────────────────
//  LED STATUS
// ─────────────────────────────────────────────────────
void updateLEDs(String status) {
  digitalWrite(LED_NORMAL,   status == "normal"   ? HIGH : LOW);
  digitalWrite(LED_WARNING,  status == "warning"  ? HIGH : LOW);
  digitalWrite(LED_CRITICAL, status == "critical" ? HIGH : LOW);
}

// ─────────────────────────────────────────────────────
//  MPU6050 I2C (manual, no lib)
// ─────────────────────────────────────────────────────
void mpu6050_init() {
  Wire.begin(21, 22);  // SDA, SCL
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_PWR_MGMT);
  Wire.write(0x00);  // Wake up
  Wire.endTransmission();
}

void mpu6050_readAccel(float &ax, float &ay, float &az) {
  Wire.beginTransmission(MPU6050_ADDR);
  Wire.write(MPU6050_ACCEL_X);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050_ADDR, 6, true);
  int16_t rawX = (Wire.read() << 8) | Wire.read();
  int16_t rawY = (Wire.read() << 8) | Wire.read();
  int16_t rawZ = (Wire.read() << 8) | Wire.read();
  float scale = 16384.0;  // ±2g range
  ax = rawX / scale;
  ay = rawY / scale;
  az = rawZ / scale;
}

// ─────────────────────────────────────────────────────
//  DEBUG PRINT
// ─────────────────────────────────────────────────────
void printReading(SensorReading& r) {
  Serial.println("─────────────────────────────────");
  Serial.printf("[SENS] Temp: %.1f°C | Vib: %.3fg | Curr: %.2fA\n",
                r.temperature, r.vibration_rms, r.current);
}
