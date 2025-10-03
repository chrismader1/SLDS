import os, pandas as pd, shutil, io
from filelock import FileLock
import tempfile

class IOManager:
    """
    All file/dir names come from CONFIG. Required CONFIG keys:
      - "tmp_dir"
      - "results_csv"
      - "segments_parquet"
      - "tmp_results_fmt"      e.g. "{tmp_dir}/results_tmp_{security}.csv"
      - "tmp_segments_fmt"     e.g. "{tmp_dir}/segments_tmp_{security}.csv"
      - "results_header_cols"  list of columns for the results CSV
      - "segments_header_cols" list of columns for segments (e.g. ["security","config","date","t","z"])
    """
    def __init__(self, CONFIG: dict):
        self.cfg = CONFIG
        # paths
        self.tmp_dir               = self._cfg("tmp_dir")
        self.results_csv           = self._cfg("results_csv")
        self.segments_parquet      = self._cfg("segments_parquet")
        self.tmp_results_fmt       = self._cfg("tmp_results_fmt")
        self.tmp_segments_fmt      = self._cfg("tmp_segments_fmt")
        # schemas
        self.results_header_cols   = list(self._cfg("results_header_cols"))
        self.segments_header_cols  = list(self._cfg("segments_header_cols"))

        # ensure dirs
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.results_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.segments_parquet), exist_ok=True)

    # ---------- cfg helper ----------
    def _cfg(self, key):
        if key not in self.cfg:
            raise KeyError(f"CONFIG missing key: {key}")
        return self.cfg[key]

    # ---------- temp file paths (per security) ----------
    def _tmp_results(self, security):
        return self.tmp_results_fmt.format(tmp_dir=self.tmp_dir, security=security)

    def _tmp_segments(self, security):
        return self.tmp_segments_fmt.format(tmp_dir=self.tmp_dir, security=security)

    # ---------- master init ----------
    def _ensure_master_results_csv(self):
        need_init = (not os.path.exists(self.results_csv)) or (os.path.getsize(self.results_csv) == 0)
        if need_init:
            pd.DataFrame(columns=self.results_header_cols).to_csv(self.results_csv, index=False)

    # ---------- temp appends (fast) ----------
    def append_temp_results(self, security, df):
        if df is None or df.empty: return
        df = df.reindex(columns=self.results_header_cols)
        tmp = self._tmp_results(security)
        write_header = (not os.path.exists(tmp)) or (os.path.getsize(tmp) == 0)
        df.to_csv(tmp, mode="a", header=write_header, index=False)

    def append_temp_segments(self, security, df):
        if df is None or df.empty: return
        df = df.reindex(columns=self.segments_header_cols)
        tmp = self._tmp_segments(security)
        write_header = (not os.path.exists(tmp)) or (os.path.getsize(tmp) == 0)
        df.to_csv(tmp, mode="a", header=write_header, index=False)
    
    # ---------- helpers ----------
    @staticmethod
    def _coerce_segments_schema(df):
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

    # ---------- flush to master ----------

    def _append_csv_file_to_master(self, tmp_csv, master_csv):
        if not (os.path.exists(tmp_csv) and os.path.getsize(tmp_csv) > 0):
            return
    
        lock = FileLock(master_csv + ".lock")
        with lock:
            new_df = pd.read_csv(tmp_csv).reindex(columns=self.results_header_cols)
            # —— SKIP if no rows ——
            if new_df.shape[0] == 0:
                return
    
            if os.path.exists(master_csv) and os.path.getsize(master_csv) > 0:
                old_df = pd.read_csv(master_csv).reindex(columns=self.results_header_cols)
                # old_df may be 0-row too; concat handles it but we avoid the warning by short-circuiting
                out = new_df if old_df.shape[0] == 0 else pd.concat([old_df, new_df], ignore_index=True)
            else:
                out = new_df
    
            d = os.path.dirname(master_csv) or "."
            with tempfile.NamedTemporaryFile("w", delete=False, dir=d, suffix=".csv") as tf:
                tmp_out = tf.name
            out.to_csv(tmp_out, index=False)
            os.replace(tmp_out, master_csv)
    
    def _append_to_parquet(self, tmp_csv, parquet_path):
        if not (os.path.exists(tmp_csv) and os.path.getsize(tmp_csv) > 0):
            return
        import pyarrow as pa, pyarrow.parquet as pq
    
        lock = FileLock(parquet_path + ".lock")
        with lock:
            new_df = pd.read_csv(tmp_csv)
            new_df = self._coerce_segments_schema(new_df)
            # —— SKIP if no rows ——
            if new_df.shape[0] == 0:
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

   def flush_one_security(self, security):
        # Always remove tmp files—even if there’s nothing to append
        tmp_res = self._tmp_results(security)
        tmp_seg = self._tmp_segments(security)
    
        # RESULTS
        if os.path.exists(tmp_res) and os.path.getsize(tmp_res) > 0:
            df = pd.read_csv(tmp_res).reindex(columns=self.results_header_cols)
            if df.shape[0] > 0:
                df.to_csv(tmp_res, index=False)
                self._append_csv_file_to_master(tmp_res, self.results_csv)
            os.remove(tmp_res)  # remove regardless
    
        # SEGMENTS
        if os.path.exists(tmp_seg) and os.path.getsize(tmp_seg) > 0:
            df = pd.read_csv(tmp_seg)
            df = self._coerce_segments_schema(df)
            if df.shape[0] > 0:
                # re-save normalized, then append
                df.to_csv(tmp_seg, index=False)
                self._append_to_parquet(tmp_seg, self.segments_parquet)
            os.remove(tmp_seg)  # remove regardless

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


