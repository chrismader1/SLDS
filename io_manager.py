# io_manager.py
import os, shutil, io
import pandas as pd
from filelock import FileLock
import tempfile
import csv

class IOManager:
    def __init__(self, CONFIG: dict):
        self.cfg = CONFIG
        self.tmp_dir               = self._cfg("tmp_dir")
        self.results_csv           = self._cfg("results_csv")
        self.segments_parquet      = self._cfg("segments_parquet")
        self.tmp_results_fmt       = self._cfg("tmp_results_fmt")
        self.tmp_segments_fmt      = self._cfg("tmp_segments_fmt")
        self.results_header_cols   = list(self._cfg("results_header_cols"))
        self.segments_header_cols  = list(self._cfg("segments_header_cols"))

        # --- centralize IO kwargs ---

        # RESULTS (write + read)
        self._WRITE_KW_RESULTS = dict(
            quoting=csv.QUOTE_MINIMAL, escapechar="\\", doublequote=True, lineterminator="\n")
        
        self._READ_KW_RESULTS = dict(
            engine="python",
            quotechar='"',
            escapechar="\\",
            doublequote=True,
            on_bad_lines="warn",)
        
        # SEGMENTS (write + read)
        self._WRITE_KW_SEG = dict(
            quoting=csv.QUOTE_MINIMAL, escapechar="\\", doublequote=True, lineterminator="\n")
        
        self._READ_KW_SEG = dict(
            engine="python", quotechar='"', escapechar="\\", doublequote=True, on_bad_lines="warn")
        
        os.makedirs(self.tmp_dir, exist_ok=True)
        res_dir = os.path.dirname(self.results_csv)
        seg_dir = os.path.dirname(self.segments_parquet)
        if res_dir:
            os.makedirs(res_dir, exist_ok=True)
        if seg_dir:
            os.makedirs(seg_dir, exist_ok=True)

    # ---------- small helpers ----------
    def _cfg(self, key):
        if key not in self.cfg:
            raise KeyError(f"CONFIG missing key: {key}")
        return self.cfg[key]

    def _tmp_results(self, security):
        return self.tmp_results_fmt.format(tmp_dir=self.tmp_dir, security=security)

    def _tmp_segments(self, security):
        return self.tmp_segments_fmt.format(tmp_dir=self.tmp_dir, security=security)

    # ---------- schema helpers ----------
    @staticmethod
    def _coerce_segments_schema(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce")
        if "t" in out.columns:
            out["t"] = pd.to_numeric(out["t"], errors="coerce").astype("Int32")
        if "z" in out.columns:
            out["z"] = pd.to_numeric(out["z"], errors="coerce").astype("Int16")
        if "security" in out.columns:
            out["security"] = out["security"].astype("string")
        if "config" in out.columns:
            out["config"] = out["config"].astype("string")
        return out
    
    def _peek_file(self, path, n=3, label="(peek)"):
        try:
            print(f"[IO][{label}] path={path} size={os.path.getsize(path)} bytes")
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for i in range(n):
                    line = f.readline()
                    if not line: break
                    print(f"[IO][{label}] L{i+1}: {line.rstrip()[:240]}")
        except Exception as e:
            print(f"[IO][{label}] <peek failed> {e}")

    # ---------- master init ----------
    def _ensure_master_results_csv(self):
        need_init = (not os.path.exists(self.results_csv)) or (os.path.getsize(self.results_csv) == 0)
        if need_init:
            print(f"[IO] init master results at {self.results_csv}")
            pd.DataFrame(columns=self.results_header_cols).to_csv(
                self.results_csv, index=False, **self._WRITE_KW_RESULTS)

    # ---------- temp appends (fast) ----------
    def append_temp_results(self, security, df):
        if df is None or df.empty:
            print(f"[IO][results] skip empty append for {security}")
            return
        df = df.reindex(columns=self.results_header_cols)
        tmp = self._tmp_results(security)
        write_header = (not os.path.exists(tmp)) or (os.path.getsize(tmp) == 0)
        print(f"[IO][results] append -> {tmp} rows={len(df)} header={write_header}")
        df.to_csv(tmp, mode="a", header=write_header, index=False, **self._WRITE_KW_RESULTS)
        self._peek_file(tmp, n=2, label=f"{security}-tmp_res")

    def append_temp_segments(self, security, df):
        if df is None or df.empty:
            print(f"[IO][segments] skip empty append for {security}")
            return
        df = df.reindex(columns=self.segments_header_cols)
        tmp = self._tmp_segments(security)
        write_header = (not os.path.exists(tmp)) or (os.path.getsize(tmp) == 0)
        print(f"[IO][segments] append -> {tmp} rows={len(df)} header={write_header}")
        df.to_csv(tmp, mode="a", header=write_header, index=False, **self._WRITE_KW_SEG)
        self._peek_file(tmp, n=2, label=f"{security}-tmp_seg")

    # ---------- flush to master ----------
    def _append_csv_file_to_master(self, tmp_csv, master_csv):
        if not (os.path.exists(tmp_csv) and os.path.getsize(tmp_csv) > 0):
            print(f"[IO][merge] nothing to append from {tmp_csv}")
            return

        # Ensure master exists with correct header
        self._ensure_master_results_csv()

        print(f"[IO][merge] start merge tmp -> master\n  tmp={tmp_csv}\n  master={master_csv}")
        self._peek_file(tmp_csv, n=2, label="tmp-read")

        lock = FileLock(master_csv + ".lock")
        with lock:
            new_df = pd.read_csv(tmp_csv, **self._READ_KW_RESULTS).reindex(columns=self.results_header_cols)
            print(f"[IO][merge] read tmp rows={new_df.shape[0]} cols={new_df.shape[1]}")

            if new_df.shape[0] == 0:
                print("[IO][merge] tmp has 0 rows; skip")
                return

            if os.path.exists(master_csv) and os.path.getsize(master_csv) > 0:
                old_df = pd.read_csv(master_csv, **self._READ_KW_RESULTS).reindex(columns=self.results_header_cols)
                print(f"[IO][merge] read master rows={old_df.shape[0]} cols={old_df.shape[1]}")
                out = new_df if old_df.shape[0] == 0 else pd.concat([old_df, new_df], ignore_index=True)
            else:
                out = new_df

            d = os.path.dirname(master_csv) or "."
            with tempfile.NamedTemporaryFile("w", delete=False, dir=d, suffix=".csv") as tf:
                tmp_out = tf.name
            out.to_csv(tmp_out, index=False, **self._WRITE_KW_RESULTS)
            os.replace(tmp_out, master_csv)
            print(f"[IO][merge] wrote master rows={out.shape[0]} -> {master_csv}")
            self._peek_file(master_csv, n=2, label="master-after")

    def _append_to_parquet(self, tmp_csv, parquet_path):
        if not (os.path.exists(tmp_csv) and os.path.getsize(tmp_csv) > 0):
            print(f"[IO][parquet] nothing to append from {tmp_csv}")
            return

        print(f"[IO][parquet] start merge tmp -> parquet\n  tmp={tmp_csv}\n  pq={parquet_path}")
        self._peek_file(tmp_csv, n=2, label="seg-tmp-read")

        import pyarrow as pa, pyarrow.parquet as pq
        lock = FileLock(parquet_path + ".lock")
        with lock:
            new_df = pd.read_csv(tmp_csv, **self._READ_KW_SEG)
            new_df = self._coerce_segments_schema(new_df)
            print(f"[IO][parquet] read tmp rows={new_df.shape[0]} cols={new_df.shape[1]}")
            if new_df.shape[0] == 0:
                print("[IO][parquet] tmp has 0 rows; skip")
                return

            if os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0:
                old = pd.read_parquet(parquet_path, engine="pyarrow")
                old = self._coerce_segments_schema(old)
                df = new_df if old.shape[0] == 0 else pd.concat([old, new_df], ignore_index=True)
            else:
                df = new_df

            d = os.path.dirname(parquet_path) or "."
            with tempfile.NamedTemporaryFile("wb", delete=False, dir=d, suffix=".parquet") as tf:
                tmp_out = tf.name
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp_out)
            os.replace(tmp_out, parquet_path)
            print(f"[IO][parquet] wrote parquet rows={df.shape[0]} -> {parquet_path}")

    def flush_one_security(self, security):
        tmp_res = self._tmp_results(security)
        tmp_seg = self._tmp_segments(security)
        print(f"[IO][flush] security={security}")
        print(f"[IO][flush] tmp_res={tmp_res} exists={os.path.exists(tmp_res)}")
        print(f"[IO][flush] tmp_seg={tmp_seg} exists={os.path.exists(tmp_seg)}")

        # RESULTS
        if os.path.exists(tmp_res) and os.path.getsize(tmp_res) > 0:
            self._peek_file(tmp_res, n=2, label=f"{security}-tmp_res-pre")
            df = pd.read_csv(tmp_res, **self._READ_KW_RESULTS).reindex(columns=self.results_header_cols)
            print(f"[IO][flush] results tmp rows={df.shape[0]} cols={df.shape[1]}")
            if df.shape[0] > 0:
                df.to_csv(tmp_res, index=False, **self._WRITE_KW_RESULTS)
                self._append_csv_file_to_master(tmp_res, self.results_csv)
            os.remove(tmp_res)
            print(f"[IO][flush] removed {tmp_res}")

        # SEGMENTS
        if os.path.exists(tmp_seg) and os.path.getsize(tmp_seg) > 0:
            self._peek_file(tmp_seg, n=2, label=f"{security}-tmp_seg-pre")
            df = pd.read_csv(tmp_seg, **self._READ_KW_SEG)
            df = self._coerce_segments_schema(df)
            print(f"[IO][flush] segments tmp rows={df.shape[0]} cols={df.shape[1]}")
            if df.shape[0] > 0:
                df.to_csv(tmp_seg, index=False, **self._WRITE_KW_SEG)
                self._append_to_parquet(tmp_seg, self.segments_parquet)
            os.remove(tmp_seg)
            print(f"[IO][flush] removed {tmp_seg}")

    # ---------- reader ----------
    def read_segments_for_stock(self, security, parquet_path=None):
        parquet_path = parquet_path or self.segments_parquet
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(parquet_path)
        df = pd.read_parquet(parquet_path)
        if "security" not in df.columns:
            raise ValueError("Parquet missing 'security' column.")
        df = self._coerce_segments_schema(df)
        return df[df["security"] == security].copy().sort_values(["config","date","t"])



