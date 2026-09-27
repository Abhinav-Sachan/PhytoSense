/*
  PhytoSense ESP32 firmware
  Pin map matches the ACTUAL wiring on the breadboard (checked 27-09-2026).

  Reads the AD8232 plant-signal amplifier, the LDR light sensor and the
  DHT11 temperature/humidity sensor, and POSTs a batch of readings once
  per second to the PhytoSense backend running on your laptop.

  Needs only one extra library: "DHT sensor library" by Adafruit.
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>
#include "secrets.h"

// ---- Pins (actual breadboard wiring) ------------------------------------
#define PIN_AD8232_OUT 34   // D34 - AD8232 OUTPUT, analog (ADC1_CH6)
#define PIN_AD8232_LOM 33   // D33 - AD8232 LO- (lead-off minus), digital input
#define PIN_AD8232_LOP 32   // D32 - AD8232 LO+ (lead-off plus),  digital input
#define PIN_LDR        35   // D35 - LDR / resistor divider, analog (ADC1_CH7)
#define PIN_DHT        27   // D27 - DHT11 data line
// AD8232 SDN is not connected: the SparkFun board then stays switched on.
// Never set LO+/LO- pins as OUTPUT - they are outputs of the AD8232.

#define DHTTYPE DHT11      // blue 4-pin sensor = DHT11 (a DHT22 is white)
DHT dht(PIN_DHT, DHTTYPE);

// ---- Timing -------------------------------------------------------------
const int SAMPLES_PER_BATCH = 10;           // 10 samples = 1 second of data per send
const int READS_PER_SAMPLE  = 100;          // each sample = average of 100 readings...
const unsigned long READ_INTERVAL_US = 1000; // ...taken 1 ms apart = 100 ms window.
                                            // Averaging over exactly 100 ms cancels
                                            // 50 Hz mains hum (5 full cycles).
const unsigned long DHT_READ_MS = 2000;     // DHT11 needs at least 1 s between reads
const unsigned long WIFI_TIMEOUT_MS = 20000;

unsigned long lastDhtTime = 0;
float lastTempC = NAN;
float lastHumidity = NAN;

// ---- Wi-Fi --------------------------------------------------------------
bool connectWiFi() {
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - start > WIFI_TIMEOUT_MS) {
      Serial.println("\nWi-Fi failed. Check name/password in secrets.h. Retrying soon...");
      return false;
    }
    delay(400);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. ESP32 IP address: ");
  Serial.println(WiFi.localIP());
  return true;
}

// ---- Sensors ------------------------------------------------------------
// One plant-signal sample: average of 100 readings over 100 ms, in millivolts.
float readBioMv() {
  unsigned long sum = 0;
  unsigned long next = micros();
  for (int i = 0; i < READS_PER_SAMPLE; i++) {
    while ((long)(micros() - next) < 0) { }   // wait for the next 1 ms slot
    next += READ_INTERVAL_US;
    sum += analogReadMilliVolts(PIN_AD8232_OUT);
  }
  return (float)sum / READS_PER_SAMPLE;
}

bool isLeadOff() {
  return digitalRead(PIN_AD8232_LOM) == HIGH || digitalRead(PIN_AD8232_LOP) == HIGH;
}

// LDR divider: more light -> higher voltage -> higher percentage.
float readLdrPercent() {
  float mv = analogReadMilliVolts(PIN_LDR);
  return (mv / 3300.0f) * 100.0f;
}

void readDhtIfDue() {
  unsigned long now = millis();
  if (now - lastDhtTime < DHT_READ_MS) return;
  lastDhtTime = now;

  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (!isnan(h) && !isnan(t)) {
    lastHumidity = h;
    lastTempC = t;
  } else {
    Serial.println("DHT11 read failed (check wiring / 4.7k pull-up)");
  }
}

// ---- Sending ------------------------------------------------------------
void sendBatch(float *samples, int count, bool leadOff) {
  // Build the JSON text by hand, so no extra library is needed
  String body = "{\"bio_mv\":[";
  for (int i = 0; i < count; i++) {
    if (i > 0) body += ",";
    body += String(samples[i], 2);
  }
  body += "]";

  if (!isnan(lastTempC))    body += ",\"temp_c\":" + String(lastTempC, 1);
  if (!isnan(lastHumidity)) body += ",\"humidity_pct\":" + String(lastHumidity, 1);
  body += ",\"light_pct\":" + String(readLdrPercent(), 1);
  body += ",\"lead_off\":" + String(leadOff ? "true" : "false");
  body += ",\"rssi\":" + String((int)WiFi.RSSI());
  body += ",\"uptime_s\":" + String(millis() / 1000.0, 1);
  body += "}";

  // Try up to 2 times: a phone hotspot sometimes drops a single request
  int code = -1;
  for (int attempt = 1; attempt <= 2; attempt++) {
    HTTPClient http;
    http.setTimeout(3000);   // don't hang if the laptop server is off
    http.begin(String(SERVER_URL) + "/api/ingest");
    http.addHeader("Content-Type", "application/json");
    http.addHeader("X-Device-Token", DEVICE_TOKEN);
    code = http.POST(body);
    http.end();
    if (code > 0) break;     // got an answer from the server, no need to retry
    delay(200);
  }

  Serial.print("POST -> ");
  Serial.print(code);
  if (code == 200)       Serial.println("  OK");
  else if (code == 401)  Serial.println("  wrong DEVICE_TOKEN in secrets.h");
  else if (code < 0)     Serial.println("  can't reach server: check SERVER_URL and that main.py is running");
  else                   Serial.println("  server error");
}

// ---- Main ---------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== PhytoSense starting ===");

  pinMode(PIN_AD8232_LOM, INPUT);
  pinMode(PIN_AD8232_LOP, INPUT);

  analogReadResolution(12);
  dht.begin();

  connectWiFi();
  Serial.println("Remember: the modified filter needs ~3-5 min to settle (wiring manual).");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    if (!connectWiFi()) {
      delay(5000);
      return;
    }
  }

  readDhtIfDue();

  float batch[SAMPLES_PER_BATCH];
  bool leadOff = false;
  for (int i = 0; i < SAMPLES_PER_BATCH; i++) {
    batch[i] = readBioMv();             // takes 100 ms
    leadOff = leadOff || isLeadOff();
  }

  sendBatch(batch, SAMPLES_PER_BATCH, leadOff);
}