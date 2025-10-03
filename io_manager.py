import os, pandas as pd, shutil, io

# io_manager.py
import os
import pandas as pd

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
        if not os.path.exists(self.results_csv):
            pd.DataFrame(columns=self.results_header_cols).to_csv(self.results_csv, index=False)

    # ---------- temp appends (fast) ----------
    def append_temp_results(self, security, df):
        if df is None or df.empty: return
        df = df.reindex(columns=self.results_header_cols)
        tmp = self._tmp_results(security)
        write_header = not os.path.exists(tmp)
        df.to_csv(tmp, mode="a", header=write_header, index=False)

    def append_temp_segments(self, security, df):
        if df is None or df.empty: return
        df = df.reindex(columns=self.segments_header_cols)
        tmp = self._tmp_segments(security)
        write_header = not os.path.exists(tmp)
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
        if not os.path.exists(tmp_csv): return
        self._ensure_master_results_csv()
        with open(tmp_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) <= 1:
            return
        with open(master_csv, "a", encoding="utf-8") as out:
            out.writelines(lines[1:])

    def _append_to_parquet(self, tmp_csv, parquet_path):
        if not os.path.exists(tmp_csv): return
        new_df = pd.read_csv(tmp_csv)
        new_df = self._coerce_segments_schema(new_df)
        if os.path.exists(parquet_path):
            old = pd.read_parquet(parquet_path)
            old = self._coerce_segments_schema(old)
            df = pd.concat([old, new_df], ignore_index=True)
        else:
            df = new_df
        df.to_parquet(parquet_path, index=False)

    def flush_one_security(self, security):
        tmp_res = self._tmp_results(security)
        tmp_seg = self._tmp_segments(security)
        self._append_csv_file_to_master(tmp_res, self.results_csv)
        self._append_to_parquet(tmp_seg, self.segments_parquet)
        for p in (tmp_res, tmp_seg):
            try: os.remove(p)
            except FileNotFoundError: pass

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


