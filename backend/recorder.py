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
  samples  : one row per sample (time, bio_mv, dc_uv, temp, humidity, light, lead_off)

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
    def add_batch(self, bio_mv_list, telemetry, dc_uv_list=None):
        """Called for every batch the ESP32 sends. Does nothing unless recording.
        The batch covers the last second, so sample i gets a time spread over it."""
        with self._lock:
            if self.active_id is None:
                return 0
            now = time.time()
            n = len(bio_mv_list)
            lead_off = telemetry.get("lead_off")
            rows = []
            for i, value in enumerate(bio_mv_list):
                t = now - (n - 1 - i) / self.sample_rate
                t = max(t, self.active_started)   # never earlier than the Start press
                dc = None
                if dc_uv_list is not None and i < len(dc_uv_list):
                    dc = dc_uv_list[i]
                rows.append((self.active_id, t, float(value), dc,
                             telemetry.get("temp_c"), telemetry.get("humidity_pct"),
                             telemetry.get("light_pct"),
                             None if lead_off is None else int(bool(lead_off))))
            with self._connect() as conn:
                conn.executemany(
                    "INSERT INTO samples (session_id, t, bio_mv, dc_uv, temp_c, humidity_pct, light_pct, lead_off) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
                conn.execute("UPDATE sessions SET sample_count = sample_count + ? WHERE id = ?",
                             (len(rows), self.active_id))
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
        writer.writerow(["session_id", "label", "t_unix", "t_rel_s", "bio_mv", "dc_uv",
                         "temp_c", "humidity_pct", "light_pct", "lead_off"])
        with self._connect() as conn:
            for r in conn.execute("SELECT * FROM samples WHERE session_id = ? ORDER BY t", (session_id,)):
                writer.writerow([session_id, session["label"], round(r["t"], 3),
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

    os.remove(test_db)
    print("\nSelf-test finished. If all 9 lines look right, recorder.py works!")