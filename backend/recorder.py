"""
PhytoSense recorder: saves labelled recording sessions to a small database.

How it works:
  - You press Start on the dashboard and pick a label (baseline / drought / ...).
  - While a session is running, every sample the ESP32 sends is saved,
    together with the temperature, humidity, light and electrode status.
  - You press Stop. The session is now part of your training dataset.

The database is ONE file: backend/data/phytosense.db (SQLite, built into Python).
It has two tables:
  sessions : one row per recording (label, note, start/end time, sample count)
  samples  : one row per sample (time, seq, bio_mv, dc_uv, temp, humidity, light, lead_off)

Exact timing: the ESP32 numbers every sample (seq = 0, 1, 2, ... since it
switched on) and measures exactly 10 per second. So sample number N was
measured at  anchor + (N + 1) * 0.1 s,  where "anchor" is the moment the
ESP32 started counting. We estimate the anchor from the batch that arrived
fastest (Wi-Fi delay only ever makes a batch LATE, never early). A jump in
seq means samples went missing, and that is counted per session.

dc_uv is empty (NULL) for now. It will hold the ADS1115 slow-channel reading
once that module is added, so the database never has to change.
"""

import contextlib
import csv
import io
import os
import sqlite3
import threading
import time

LABELS = ["baseline", "drought", "mechanical", "test"]

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "phytosense.db")


class Recorder:
    def __init__(self, db_path=DEFAULT_DB_PATH, sample_rate=10):
        self.db_path = db_path
        self.sample_rate = sample_rate
        self._lock = threading.Lock()   # Flask can call us from several threads at once
        self.active_id = None           # id of the session being recorded, or None
        self.active_started = None      # when that session started (unix time)

        # Device clock (see top of file). Reset whenever the ESP32 restarts.
        self.clock_anchor = None        # unix time when the ESP32 had seq = 0
        self.last_seq_seen = None       # newest seq received (recording or not)
        self.last_uptime = None         # ESP32 uptime in the last batch (drops when it restarts)
        self.last_saved_seq = None      # newest seq saved in the running session

        folder = os.path.dirname(db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self._create_tables()
        self._close_unfinished_sessions()

    # ---- database helpers ------------------------------------------------
    @contextlib.contextmanager
    def _connect(self):
        """Open the database, save changes if all went well, and always close it
        (Windows can't delete or move a file that is still open)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row   # lets us read columns by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _create_tables(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    label        TEXT NOT NULL,
                    note         TEXT,
                    started_at   REAL NOT NULL,
                    ended_at     REAL,
                    sample_count INTEGER NOT NULL DEFAULT 0,
                    sample_rate  REAL NOT NULL
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS samples (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   INTEGER NOT NULL REFERENCES sessions(id),
                    t            REAL NOT NULL,
                    bio_mv       REAL NOT NULL,
                    dc_uv        REAL,
                    temp_c       REAL,
                    humidity_pct REAL,
                    light_pct    REAL,
                    lead_off     INTEGER
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_session ON samples(session_id)")

            # Columns added in a later version: add them to older databases too
            sample_cols = [r["name"] for r in conn.execute("PRAGMA table_info(samples)")]
            if "seq" not in sample_cols:
                conn.execute("ALTER TABLE samples ADD COLUMN seq INTEGER")
            session_cols = [r["name"] for r in conn.execute("PRAGMA table_info(sessions)")]
            if "missing_samples" not in session_cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN missing_samples INTEGER NOT NULL DEFAULT 0")

    def _close_unfinished_sessions(self):
        """If the server was stopped mid-recording, close that session properly
        so it doesn't stay 'recording' forever."""
        with self._connect() as conn:
            open_rows = conn.execute("SELECT id, started_at FROM sessions WHERE ended_at IS NULL").fetchall()
            for row in open_rows:
                last_t = conn.execute("SELECT MAX(t) FROM samples WHERE session_id = ?", (row["id"],)).fetchone()[0]
                conn.execute("UPDATE sessions SET ended_at = ?, note = COALESCE(note, '') || ' [auto-closed after server restart]' WHERE id = ?",
                             (last_t or row["started_at"], row["id"]))

    # ---- start / stop ----------------------------------------------------
    def start(self, label, note=""):
        """Begin a new session. Returns (session_dict, error_message)."""
        label = (label or "").strip().lower()
        if label not in LABELS:
            return None, f"label must be one of {LABELS}"
        with self._lock:
            if self.active_id is not None:
                return None, "a recording is already running - stop it first"
            started = time.time()
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO sessions (label, note, started_at, sample_rate) VALUES (?, ?, ?, ?)",
                    (label, (note or "").strip(), started, self.sample_rate))
                self.active_id = cur.lastrowid
                self.active_started = started
                self.last_saved_seq = None
        return self.get_session(self.active_id), None

    def stop(self):
        """Finish the running session. Returns (session_dict, error_message)."""
        with self._lock:
            if self.active_id is None:
                return None, "no recording is running"
            session_id = self.active_id
            with self._connect() as conn:
                conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time(), session_id))
            self.active_id = None
            self.active_started = None
        return self.get_session(session_id), None

    # ---- saving samples --------------------------------------------------
    def _update_clock(self, seq_list, uptime, now):
        """Keep track of when the ESP32 started counting (see top of file)."""
        period = 1.0 / self.sample_rate
        restarted = False
        if uptime is not None and self.last_uptime is not None and uptime < self.last_uptime - 1:
            restarted = True     # uptime went backwards -> the ESP32 restarted, seq begins at 0 again
        elif uptime is None and self.last_seq_seen is not None and seq_list[0] < self.last_seq_seen - 1000:
            restarted = True     # (no uptime sent) a big jump backwards also means a restart
        if restarted:
            self.clock_anchor = None
            self.last_saved_seq = None
        # A small step backwards is NOT a restart: the ESP32 re-sent a batch whose
        # reply got lost. Those samples are skipped below as already saved.
        if uptime is not None:
            self.last_uptime = uptime
        # The newest sample in this batch finished measuring no later than "now"
        candidate = now - (seq_list[-1] + 1) * period
        if self.clock_anchor is None or candidate < self.clock_anchor:
            self.clock_anchor = candidate    # the fastest-arriving batch is the most accurate
        self.last_seq_seen = max(seq_list[-1], self.last_seq_seen or 0) if not restarted else seq_list[-1]

    def add_batch(self, bio_mv_list, telemetry, dc_uv_list=None, seq_list=None, now=None):
        """Called for every batch the ESP32 sends. Saves it only while recording.

        With seq_list (firmware v2.1+): exact times from the sample numbers,
        samples measured before Start are skipped, and gaps are counted.
        Without seq_list (older firmware): times are spread over the last second.
        ("now" is only passed in by the self-test below.)"""
        with self._lock:
            now = time.time() if now is None else now
            period = 1.0 / self.sample_rate
            if seq_list is not None:
                self._update_clock(seq_list, telemetry.get("uptime_s"), now)

            if self.active_id is None:
                return 0

            n = len(bio_mv_list)
            lead_off = telemetry.get("lead_off")
            lead_off_value = None if lead_off is None else int(bool(lead_off))
            rows = []
            missing = 0

            for i, value in enumerate(bio_mv_list):
                seq = None
                if seq_list is not None:
                    seq = seq_list[i]
                    t = self.clock_anchor + (seq + 1) * period
                    if t < self.active_started:
                        continue            # measured before Start was pressed -> not part of this session
                    if self.last_saved_seq is not None:
                        if seq <= self.last_saved_seq:
                            continue        # already saved (the ESP32 re-sent a batch)
                        missing += seq - self.last_saved_seq - 1
                    self.last_saved_seq = seq
                else:
                    t = now - (n - 1 - i) * period
                    t = max(t, self.active_started)   # never earlier than the Start press

                dc = None
                if dc_uv_list is not None and i < len(dc_uv_list):
                    dc = dc_uv_list[i]
                rows.append((self.active_id, t, seq, float(value), dc,
                             telemetry.get("temp_c"), telemetry.get("humidity_pct"),
                             telemetry.get("light_pct"), lead_off_value))

            if not rows:
                return 0
            with self._connect() as conn:
                conn.executemany(
                    "INSERT INTO samples (session_id, t, seq, bio_mv, dc_uv, temp_c, humidity_pct, light_pct, lead_off) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
                conn.execute("UPDATE sessions SET sample_count = sample_count + ?, "
                             "missing_samples = missing_samples + ? WHERE id = ?",
                             (len(rows), missing, self.active_id))
            return len(rows)

    # ---- reading back ----------------------------------------------------
    def get_session(self, session_id):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            lead_off_count = conn.execute(
                "SELECT COUNT(*) FROM samples WHERE session_id = ? AND lead_off = 1", (session_id,)).fetchone()[0]
        s = dict(row)
        end = s["ended_at"] or time.time()
        s["duration_s"] = round(end - s["started_at"], 1)
        s["recording"] = s["ended_at"] is None
        s["lead_off_samples"] = lead_off_count
        return s

    def status(self):
        """What the dashboard shows: is a recording running, and how far along."""
        if self.active_id is None:
            return {"recording": False, "session": None, "labels": LABELS}
        return {"recording": True, "session": self.get_session(self.active_id), "labels": LABELS}

    def list_sessions(self, limit=50):
        with self._connect() as conn:
            ids = [r["id"] for r in conn.execute(
                "SELECT id FROM sessions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()]
        return [self.get_session(i) for i in ids]

    def export_csv(self, session_id):
        """Returns the session's samples as CSV text, or None if it doesn't exist."""
        session = self.get_session(session_id)
        if session is None:
            return None
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["session_id", "label", "seq", "t_unix", "t_rel_s", "bio_mv", "dc_uv",
                         "temp_c", "humidity_pct", "light_pct", "lead_off"])
        with self._connect() as conn:
            for r in conn.execute("SELECT * FROM samples WHERE session_id = ? ORDER BY t, id", (session_id,)):
                writer.writerow([session_id, session["label"], r["seq"], round(r["t"], 3),
                                 round(r["t"] - session["started_at"], 3), r["bio_mv"], r["dc_uv"],
                                 r["temp_c"], r["humidity_pct"], r["light_pct"], r["lead_off"]])
        return out.getvalue()

    def delete_session(self, session_id):
        """Remove a bad recording (e.g. electrodes fell off). Can't delete the running one."""
        with self._lock:
            if session_id == self.active_id:
                return False, "stop the recording before deleting it"
            with self._connect() as conn:
                conn.execute("DELETE FROM samples WHERE session_id = ?", (session_id,))
                cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            if cur.rowcount == 0:
                return False, "session not found"
        return True, None


# ---- Self-test: run  python recorder.py  to check everything works -------
if __name__ == "__main__":
    import tempfile

    test_db = os.path.join(tempfile.gettempdir(), "phytosense_selftest.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    rec = Recorder(db_path=test_db)
    print("1. Nothing recording:", rec.status()["recording"])

    print("2. Batch while NOT recording is ignored ->", rec.add_batch([1650.0] * 10, {}), "rows saved")

    session, err = rec.start("baseline", "self-test")
    print("3. Started session", session["id"], "label =", session["label"])

    _, err = rec.start("drought")
    print("4. Second start refused:", err)

    fake_telemetry = {"temp_c": 25.5, "humidity_pct": 80.0, "light_pct": 50.0, "lead_off": False}
    for _ in range(3):
        rec.add_batch([1650.0 + i for i in range(10)], fake_telemetry)
    print("5. Samples saved:", rec.status()["session"]["sample_count"], "(expected 30)")

    session, _ = rec.stop()
    print("6. Stopped. Duration:", session["duration_s"], "s, samples:", session["sample_count"])

    csv_text = rec.export_csv(session["id"])
    print("7. CSV has", len(csv_text.strip().splitlines()) - 1, "data rows. First two lines:")
    print("   " + "\n   ".join(csv_text.splitlines()[:2]))

    _, err = rec.start("sunshine")
    print("8. Bad label refused:", err)

    ok, _ = rec.delete_session(session["id"])
    print("9. Deleted:", ok, "- sessions left:", len(rec.list_sessions()))

    # Exact timing with sample numbers (firmware v2.1): seq 0..9, then 10..19,
    # then a batch with 5 missing samples (seq 25..34) that arrives late.
    t0 = time.time()
    fake_telemetry = dict(fake_telemetry, uptime_s=100.0)
    rec.add_batch([1.0] * 10, fake_telemetry, seq_list=list(range(0, 10)), now=t0)   # before Start: skipped
    session, _ = rec.start("test", "seq test")
    rec.add_batch([2.0] * 10, fake_telemetry, seq_list=list(range(10, 20)), now=t0 + 1.05)
    rec.add_batch([3.0] * 10, fake_telemetry, seq_list=list(range(25, 35)), now=t0 + 2.9)   # late + 5 missing
    rec.add_batch([3.0] * 10, fake_telemetry, seq_list=list(range(25, 35)), now=t0 + 3.0)   # duplicate: ignored
    session, _ = rec.stop()
    rows = rec.export_csv(session["id"]).strip().splitlines()[1:]
    steps = {round(float(b.split(",")[4]) - float(a.split(",")[4]), 3) for a, b in zip(rows, rows[1:])}
    print("10. With sample numbers: saved", session["sample_count"], "(expected 20), missing",
          session["missing_samples"], "(expected 5), time steps", sorted(steps), "(expected [0.1, 0.6])")

    # ESP32 restart during a session: uptime drops and seq starts again from 0
    session, _ = rec.start("test", "restart test")
    t1 = time.time()
    rec.add_batch([1.0] * 10, dict(fake_telemetry, uptime_s=501.0), seq_list=list(range(5000, 5010)), now=t1 + 1.0)
    rec.add_batch([1.0] * 10, dict(fake_telemetry, uptime_s=1.0), seq_list=list(range(0, 10)), now=t1 + 2.5)
    session, _ = rec.stop()
    print("11. After an ESP32 restart: saved", session["sample_count"], "(expected 20), missing",
          session["missing_samples"], "(expected 0)")

    os.remove(test_db)
    print("\nSelf-test finished. If all 11 lines look right, recorder.py works!")