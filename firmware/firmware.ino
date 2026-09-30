/*
  PhytoSense ESP32 firmware  (v2.1: non-stop sampling + sample numbers + ADS1115 ready)
  Pin map matches the ACTUAL wiring on the breadboard (checked 27-09-2026).

  How it works - the ESP32 has TWO processor cores, and we use both:
    Core 0  "sampler"  : measures the plant 10 times per second, NON-STOP.
                         It never waits for Wi-Fi, so there are no blind gaps.
    Core 1  "sender"   : (the normal loop) reads the DHT11 and sends the
                         collected samples to the laptop over Wi-Fi.
  The two cores pass samples through a queue (a waiting line in memory).
  If Wi-Fi is slow for a moment, samples wait in the queue and are sent
  a little later - nothing is lost.
  Every sample gets a number (seq = 0, 1, 2, ... since switch-on). The laptop
  uses these numbers to give each sample its exact time and to notice if
  any sample ever goes missing.

  ADS1115 (16-bit ADC for the slow DC plant signal) is OPTIONAL:
    - Not connected -> works exactly like before (bio_mv only).
    - Connected (SDA=D21, SCL=D22, ADDR=GND) -> found automatically at start-up
      and also sends "dc_uv" (electrode voltage difference A0-A1, microvolts).
  Always unplug the USB before connecting or disconnecting the ADS1115.

  Libraries needed (Arduino IDE -> Library Manager):
    "DHT sensor library" by Adafruit       (already installed)
    "Adafruit ADS1X15"   by Adafruit       (new - click INSTALL ALL)
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <DHT.h>
#include <Adafruit_ADS1X15.h>
#include "secrets.h"

// ---- Pins (actual breadboard wiring) ------------------------------------
#define PIN_AD8232_OUT 34   // D34 - AD8232 OUTPUT, analog (ADC1_CH6)
#define PIN_AD8232_LOM 33   // D33 - AD8232 LO- (lead-off minus), digital input
#define PIN_AD8232_LOP 32   // D32 - AD8232 LO+ (lead-off plus),  digital input
#define PIN_LDR        35   // D35 - LDR / resistor divider, analog (ADC1_CH7)
#define PIN_DHT        27   // D27 - DHT11 data line
#define PIN_SDA        21   // D21 - ADS1115 SDA (I2C data)
#define PIN_SCL        22   // D22 - ADS1115 SCL (I2C clock)
// AD8232 SDN is not connected: the SparkFun board then stays switched on.
// Never set LO+/LO- pins as OUTPUT - they are outputs of the AD8232.

#define DHTTYPE DHT11      // blue 4-pin sensor = DHT11 (a DHT22 is white)
DHT dht(PIN_DHT, DHTTYPE);

Adafruit_ADS1115 ads;
const uint8_t ADS_ADDRESS = 0x48;   // ADDR pin connected to GND
bool adsFound = false;

// ---- Timing -------------------------------------------------------------
// One sample = average of 100 AD8232 readings taken 1 ms apart = 100 ms.
// Averaging over exactly 100 ms cancels 50 Hz mains hum (5 full cycles).
// 10 samples per second, continuously.
const int READS_PER_SAMPLE = 100;
const int ADS_EVERY_N_READS = 4;     // ADS1115 runs at 250/s = one new value every 4 ms
const unsigned long DHT_READ_MS = 2000;     // DHT11 needs at least 1 s between reads
const unsigned long WIFI_TIMEOUT_MS = 20000;

// ---- Sending ------------------------------------------------------------
const int MIN_SAMPLES_TO_SEND = 10;   // normally send once per second (10 samples)
const int MAX_SAMPLES_PER_POST = 50;  // after a Wi-Fi hiccup, catch up 5 s at a time
const int QUEUE_LENGTH = 300;         // up to 30 s of samples can wait for Wi-Fi

// One measurement, passed from the sampler (core 0) to the sender (core 1)
struct Sample {
  uint32_t seq;     // sample number since the ESP32 switched on (10 per second)
  float bioMv;      // AD8232 output, millivolts
  float dcUv;       // ADS1115 A0-A1 difference, microvolts (NAN if no ADS1115)
  float ldrMv;      // LDR divider voltage, millivolts
  bool  leadOff;    // true = an electrode is not touching
};

QueueHandle_t sampleQueue;
volatile unsigned long droppedSamples = 0;   // only grows if Wi-Fi is down > 30 s

Sample pending[MAX_SAMPLES_PER_POST];   // samples taken out of the queue, not yet sent
int pendingCount = 0;

unsigned long lastDhtTime = 0;
float lastTempC = NAN;
float lastHumidity = NAN;

// ---- Core 0: the sampler task (runs forever) ----------------------------
void samplerTask(void *parameter) {
  TickType_t lastWake = xTaskGetTickCount();
  uint32_t nextSeq = 0;

  while (true) {
    uint32_t bioSum = 0;
    float dcSum = 0;
    int dcCount = 0;
    bool leadOff = false;

    for (int i = 0; i < READS_PER_SAMPLE; i++) {
      vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(1));   // wait for the next 1 ms slot
      bioSum += analogReadMilliVolts(PIN_AD8232_OUT);

      if (adsFound && (i % ADS_EVERY_N_READS == 0)) {
        int16_t raw = ads.getLastConversionResults();
        dcSum += ads.computeVolts(raw) * 1e6f;          // volts -> microvolts
        dcCount++;
      }
    }

    // Once per sample: electrode check + light level
    leadOff = digitalRead(PIN_AD8232_LOM) == HIGH || digitalRead(PIN_AD8232_LOP) == HIGH;

    Sample s;
    s.seq = nextSeq++;
    s.bioMv = (float)bioSum / READS_PER_SAMPLE;
    s.dcUv = (dcCount > 0) ? dcSum / dcCount : NAN;
    s.ldrMv = analogReadMilliVolts(PIN_LDR);
    s.leadOff = leadOff;

    // Put it in the queue. If the queue is full (Wi-Fi down > 30 s), drop the oldest.
    if (xQueueSend(sampleQueue, &s, 0) != pdTRUE) {
      Sample oldest;
      xQueueReceive(sampleQueue, &oldest, 0);
      xQueueSend(sampleQueue, &s, 0);
      droppedSamples++;
    }
  }
}

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

// ---- I2C + ADS1115 --------------------------------------------------------
void scanI2C() {
  Serial.print("I2C scan (SDA=D21, SCL=D22): ");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("0x");
      Serial.print(addr, HEX);
      Serial.print(" ");
      found++;
    }
  }
  if (found == 0) Serial.print("no devices");
  Serial.println();
}

void setupADS1115() {
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);   // fast I2C so each read takes only ~0.1 ms
  scanI2C();

  if (!ads.begin(ADS_ADDRESS, &Wire)) {
    Serial.println("ADS1115: not found -> sending AD8232 data only (this is fine).");
    adsFound = false;
    return;
  }

  ads.setGain(GAIN_SIXTEEN);                 // +/-0.256 V range, 7.8 uV per step
  ads.setDataRate(RATE_ADS1115_250SPS);      // 250 readings per second
  ads.startADCReading(ADS1X15_REG_CONFIG_MUX_DIFF_0_1, /*continuous=*/true);  // A0 minus A1
  adsFound = true;
  Serial.println("ADS1115: FOUND at 0x48 -> also sending dc_uv (A0-A1, +/-256 mV range).");
}

// ---- Sensors on core 1 ----------------------------------------------------
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
String buildJson() {
  String body = "{\"bio_mv\":[";
  for (int i = 0; i < pendingCount; i++) {
    if (i > 0) body += ",";
    body += String(pending[i].bioMv, 2);
  }
  body += "]";

  body += ",\"seq\":[";
  for (int i = 0; i < pendingCount; i++) {
    if (i > 0) body += ",";
    body += String(pending[i].seq);
  }
  body += "]";

  if (adsFound) {
    body += ",\"dc_uv\":[";
    for (int i = 0; i < pendingCount; i++) {
      if (i > 0) body += ",";
      if (isnan(pending[i].dcUv)) body += "null";
      else body += String(pending[i].dcUv, 1);
    }
    body += "]";
  }

  bool leadOff = false;
  for (int i = 0; i < pendingCount; i++) leadOff = leadOff || pending[i].leadOff;

  float ldrMv = pending[pendingCount - 1].ldrMv;   // newest light reading
  if (!isnan(lastTempC))    body += ",\"temp_c\":" + String(lastTempC, 1);
  if (!isnan(lastHumidity)) body += ",\"humidity_pct\":" + String(lastHumidity, 1);
  body += ",\"light_pct\":" + String(ldrMv / 3300.0f * 100.0f, 1);
  body += ",\"lead_off\":" + String(leadOff ? "true" : "false");
  body += ",\"ads1115\":" + String(adsFound ? "true" : "false");
  body += ",\"rssi\":" + String((int)WiFi.RSSI());
  body += ",\"uptime_s\":" + String(millis() / 1000.0, 1);
  body += "}";
  return body;
}

// Returns the HTTP code (200 = OK). Tries twice: a hotspot sometimes drops one request.
int sendPending() {
  String body = buildJson();
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
  return code;
}

void printResult(int code, int sent) {
  Serial.print("POST ");
  Serial.print(sent);
  Serial.print(" samples -> ");
  Serial.print(code);
  if (code == 200)       Serial.print("  OK");
  else if (code == 401)  Serial.print("  wrong DEVICE_TOKEN in secrets.h");
  else if (code < 0)     Serial.print("  can't reach server: check SERVER_URL and that main.py is running");
  else                   Serial.print("  server error");
  Serial.print("   (waiting: ");
  Serial.print(uxQueueMessagesWaiting(sampleQueue));
  if (droppedSamples > 0) {
    Serial.print(", dropped so far: ");
    Serial.print(droppedSamples);
  }
  Serial.println(")");
}

// ---- Main ---------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== PhytoSense v2.1 starting ===");

  pinMode(PIN_AD8232_LOM, INPUT);
  pinMode(PIN_AD8232_LOP, INPUT);
  analogReadResolution(12);
  dht.begin();

  setupADS1115();   // must happen BEFORE the sampler starts using it

  sampleQueue = xQueueCreate(QUEUE_LENGTH, sizeof(Sample));
  // Start the sampler on core 0 (the normal loop runs on core 1)
  xTaskCreatePinnedToCore(samplerTask, "sampler", 4096, NULL, 5, NULL, 0);
  Serial.println("Sampler running on core 0: 10 samples/s, non-stop.");

  connectWiFi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    if (!connectWiFi()) {
      delay(5000);   // the sampler keeps measuring meanwhile; samples wait in the queue
      return;
    }
  }

  readDhtIfDue();

  // Move waiting samples from the queue into our "pending" list
  Sample s;
  while (pendingCount < MAX_SAMPLES_PER_POST && xQueueReceive(sampleQueue, &s, 0) == pdTRUE) {
    pending[pendingCount++] = s;
  }

  if (pendingCount < MIN_SAMPLES_TO_SEND) {
    delay(20);        // not a full second yet - check again shortly
    return;
  }

  int sent = pendingCount;
  int code = sendPending();
  printResult(code, sent);

  if (code == 200 || code == 400 || code == 401) {
    pendingCount = 0;   // delivered (or the server will never accept it) -> clear
  }
  // Any other failure: keep the pending samples and try again next loop.
  // If Wi-Fi stays down, the queue fills up and the oldest samples are dropped.
}