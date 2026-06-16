import ast
import csv
import re
import tkinter as tk
from collections import defaultdict
from math import ceil, floor
from pathlib import Path
from statistics import mean
from tkinter import messagebox, ttk

DEFAULT_WORKER_LOGS = r"\\Moderskipet\Darsia_Queue\Kalibrering\worker_logs"
DEFAULT_RUN_LOGS = r"C:\Users\olav_\Documents\GitHub\ff_ac\logs"
DEFAULT_OUT_CSV = (
    r"C:\Users\olav_\Documents\GitHub\ff_ac\logs\worker_stats_summary.csv"
)


START_RE = re.compile(r"^\[(?P<worker>[A-Za-z0-9_]+)\] start task=(?P<task>\S+)")
_CACHE_RE = re.compile(r"\bcache=(\d+)")
_WORKERS_RE = re.compile(r"\bworkers=\d+/(\d+)")
DONE_RE = re.compile(
    r"^\[(?P<worker>[A-Za-z0-9_]+)\] done task=(?P<task>\S+) status=(?P<status>\S+)"
    r" runtime=(?P<rt>[0-9.]+)s(?: max_rss=(?P<rss>[0-9.]+)MB)?"
)
FAIL_RE = re.compile(
    r"^\[(?P<worker>[A-Za-z0-9_]+)\] failed task=(?P<task>\S+) runtime=(?P<rt>[0-9.]+)s"
)


def percentile(vals, p):
    if not vals:
        return 0.0
    v = sorted(vals)
    k = (len(v) - 1) * p
    f = floor(k)
    c = ceil(k)
    if f == c:
        return v[int(k)]
    return v[f] + (v[c] - v[f]) * (k - f)


def run_from_task(task):
    return task.split("_")[0] if "_" in task else "unknown"


def tput_per_worker(avg_rt):
    return 3600.0 / avg_rt if avg_rt > 0 else 0.0


def metric_value(entry, metric, run_images):
    rt = entry["rt"]
    workers = entry["workers"] or 0
    if metric == "avg_runtime_s":
        return rt
    if metric == "tput_per_worker_h":
        return tput_per_worker(rt)
    if metric == "tput_total_h":
        return tput_per_worker(rt) * workers
    if metric == "img_tput_total_h":
        images = run_images.get(entry["run"], 0)
        return images * tput_per_worker(rt) * workers
    if metric == "rss_mb":
        return entry["rss"] or 0.0
    return 0.0


def smooth_series(values, window):
    if window <= 1 or len(values) <= 1:
        return values
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        smoothed.append(mean(chunk))
    return smoothed


def task_ts_ms(task, fallback_ms):
    parts = task.split("_")
    if len(parts) >= 4 and parts[3].isdigit():
        return int(parts[3])
    return fallback_ms


def load_run_images(run_logs_dir):
    run_images = {}
    for csv_path in run_logs_dir.glob("auto_calibration_*.csv"):
        run = csv_path.stem.replace("auto_calibration_", "")
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics = row.get("metrics")
                    if not metrics:
                        continue
                    try:
                        m = ast.literal_eval(metrics)
                        if isinstance(m, dict):
                            run_images[run] = len(m)
                            break
                    except Exception:
                        continue
        except Exception:
            continue
    return run_images


def parse_worker_logs(worker_logs_dir):
    task_cfg = {}
    entries = []
    for path in worker_logs_dir.glob("*.log"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        file_ms = int(path.stat().st_mtime * 1000)
        for line in text.splitlines():
            m = START_RE.match(line)
            if m:
                task = m.group("task")
                worker = m.group("worker")
                host = worker.split("_")[0]
                _cm = _CACHE_RE.search(line)
                cache = int(_cm.group(1)) if _cm else None
                _wm = _WORKERS_RE.search(line)
                workers = int(_wm.group(1)) if _wm else None
                task_cfg[task] = (host, cache, worker, workers)
                continue
            m = DONE_RE.match(line)
            if m:
                task = m.group("task")
                worker = m.group("worker")
                host = worker.split("_")[0]
                cache = task_cfg.get(task, (host, None, worker, None))[1]
                workers = task_cfg.get(task, (host, None, worker, None))[3]
                entries.append(
                    {
                        "task": task,
                        "run": run_from_task(task),
                        "host": host,
                        "worker": worker,
                        "cache": cache,
                        "workers": workers,
                        "status": m.group("status"),
                        "rt": float(m.group("rt")),
                        "rss": float(m.group("rss")) if m.group("rss") else None,
                        "ts": task_ts_ms(task, file_ms),
                    }
                )
                continue
            m = FAIL_RE.match(line)
            if m:
                task = m.group("task")
                worker = m.group("worker")
                host = worker.split("_")[0]
                cache = task_cfg.get(task, (host, None, worker, None))[1]
                workers = task_cfg.get(task, (host, None, worker, None))[3]
                entries.append(
                    {
                        "task": task,
                        "run": run_from_task(task),
                        "host": host,
                        "worker": worker,
                        "cache": cache,
                        "workers": workers,
                        "status": "fail",
                        "rt": float(m.group("rt")),
                        "rss": None,
                        "ts": task_ts_ms(task, file_ms),
                    }
                )
    return entries


def compute_rows(
    entries,
    run_images,
    mode,
    selected_runs,
    selected_hosts,
    selected_caches,
    selected_workers,
    last_n,
):
    per_run = defaultdict(lambda: {"rt": [], "rss": [], "ok": 0, "total": 0, "img_sum": 0})
    overall = defaultdict(lambda: {"rt": [], "rss": [], "ok": 0, "total": 0, "img_sum": 0})

    filtered = []
    for e in entries:
        if selected_runs and e["run"] not in selected_runs:
            continue
        if selected_hosts and e["host"] not in selected_hosts:
            continue
        filtered.append(e)

    if last_n and last_n > 0:
        filtered.sort(key=lambda e: e["ts"])
        filtered = filtered[-last_n:]

    if mode == "per_run":
        avail_cache = defaultdict(set)
        avail_workers = defaultdict(set)
        for e in filtered:
            key = (e["run"], e["host"])
            if e["cache"] is not None:
                avail_cache[key].add(e["cache"])
            if e["workers"] is not None:
                avail_workers[key].add(e["workers"])
    else:
        avail_cache = defaultdict(set)
        avail_workers = defaultdict(set)
        for e in filtered:
            key = e["host"]
            if e["cache"] is not None:
                avail_cache[key].add(e["cache"])
            if e["workers"] is not None:
                avail_workers[key].add(e["workers"])

    for e in filtered:
        if mode == "per_run":
            group = (e["run"], e["host"])
        else:
            group = e["host"]

        if selected_caches:
            if avail_cache[group] & set(selected_caches):
                if e["cache"] not in selected_caches:
                    continue

        if selected_workers:
            if avail_workers[group] & set(selected_workers):
                if e["workers"] not in selected_workers:
                    continue

        images = run_images.get(e["run"], 0)
        if mode == "per_run":
            key = (e["run"], e["host"], e["workers"], e["cache"])
            per_run[key]["total"] += 1
            if e["status"] == "ok":
                per_run[key]["ok"] += 1
                per_run[key]["rt"].append(e["rt"])
                if e["rss"] is not None:
                    per_run[key]["rss"].append(e["rss"])
                per_run[key]["img_sum"] += images
        else:
            key = (e["host"], e["workers"], e["cache"])
            overall[key]["total"] += 1
            if e["status"] == "ok":
                overall[key]["ok"] += 1
                overall[key]["rt"].append(e["rt"])
                if e["rss"] is not None:
                    overall[key]["rss"].append(e["rss"])
                overall[key]["img_sum"] += images

    rows = []
    if mode == "per_run":
        for (run, host, workers, cache), s in sorted(
            per_run.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or -1, kv[0][3] or -1)
        ):
            worker_setting = workers if workers is not None else 0
            avg_rt = mean(s["rt"]) if s["rt"] else 0.0
            tput = tput_per_worker(avg_rt)
            tput_total = tput * worker_setting
            avg_images = (s["img_sum"] / s["ok"]) if s["ok"] else 0.0
            img_tput = avg_images * tput
            img_tput_total = img_tput * worker_setting
            rss_max = max(s["rss"]) if s["rss"] else 0.0
            rss_p50 = percentile(s["rss"], 0.50)
            rss_p95 = percentile(s["rss"], 0.95)
            rows.append(
                {
                    "run": run,
                    "host": host,
                    "workers": worker_setting if worker_setting else "?",
                    "cache": cache if cache is not None else "?",
                    "ok_total": f"{s['ok']}/{s['total']}",
                    "avg_runtime_s": avg_rt,
                    "tput_per_worker_h": tput,
                    "tput_total_h": tput_total,
                    "img_tput_per_worker_h": img_tput,
                    "img_tput_total_h": img_tput_total,
                    "rss_max_mb": rss_max,
                    "rss_p50_mb": rss_p50,
                    "rss_p95_mb": rss_p95,
                }
            )
    else:
        for (host, workers, cache), s in sorted(
            overall.items(), key=lambda kv: (kv[0][0], kv[0][1] or -1, kv[0][2] or -1)
        ):
            worker_setting = workers if workers is not None else 0
            avg_rt = mean(s["rt"]) if s["rt"] else 0.0
            tput = tput_per_worker(avg_rt)
            tput_total = tput * worker_setting
            avg_images = (s["img_sum"] / s["ok"]) if s["ok"] else 0.0
            img_tput = avg_images * tput
            img_tput_total = img_tput * worker_setting
            rss_max = max(s["rss"]) if s["rss"] else 0.0
            rss_p50 = percentile(s["rss"], 0.50)
            rss_p95 = percentile(s["rss"], 0.95)
            rows.append(
                {
                    "run": "*",
                    "host": host,
                    "workers": worker_setting if worker_setting else "?",
                    "cache": cache if cache is not None else "?",
                    "ok_total": f"{s['ok']}/{s['total']}",
                    "avg_runtime_s": avg_rt,
                    "tput_per_worker_h": tput,
                    "tput_total_h": tput_total,
                    "img_tput_per_worker_h": img_tput,
                    "img_tput_total_h": img_tput_total,
                    "rss_max_mb": rss_max,
                    "rss_p50_mb": rss_p50,
                    "rss_p95_mb": rss_p95,
                }
            )
    return rows


def compute_plot_series(
    entries,
    run_images,
    group_by,
    selected_runs,
    selected_hosts,
    selected_caches,
    selected_workers,
    metric,
    last_n,
):
    filtered = []
    for e in entries:
        if e["status"] != "ok":
            continue
        if selected_runs and e["run"] not in selected_runs:
            continue
        if selected_hosts and e["host"] not in selected_hosts:
            continue
        filtered.append(e)

    if last_n and last_n > 0:
        filtered.sort(key=lambda e: e["ts"])
        filtered = filtered[-last_n:]

    avail_cache = defaultdict(set)
    avail_workers = defaultdict(set)
    for e in filtered:
        if group_by == "host":
            key = e["host"]
        elif group_by == "run":
            key = e["run"]
        elif group_by == "host_run":
            key = (e["host"], e["run"])
        else:
            key = "all"
        if e["cache"] is not None:
            avail_cache[key].add(e["cache"])
        if e["workers"] is not None:
            avail_workers[key].add(e["workers"])

    series = defaultdict(list)
    for e in filtered:
        if group_by == "host":
            key = e["host"]
        elif group_by == "run":
            key = e["run"]
        elif group_by == "host_run":
            key = (e["host"], e["run"])
        else:
            key = "all"

        if selected_caches:
            if avail_cache[key] & set(selected_caches):
                if e["cache"] not in selected_caches:
                    continue

        if selected_workers:
            if avail_workers[key] & set(selected_workers):
                if e["workers"] not in selected_workers:
                    continue

        value = metric_value(e, metric, run_images)
        series[key].append((e["ts"], value))

    for key, values in series.items():
        values.sort(key=lambda x: x[0])
        series[key] = [v for _t, v in values]
    return series


class StatsUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Worker Stats Summary")
        self.geometry("1200x700")

        self.worker_logs_var = tk.StringVar(value=DEFAULT_WORKER_LOGS)
        self.run_logs_var = tk.StringVar(value=DEFAULT_RUN_LOGS)
        self.out_csv_var = tk.StringVar(value=DEFAULT_OUT_CSV)
        self.mode_var = tk.StringVar(value="per_run")
        self.last_n_var = tk.IntVar(value=0)
        self.smooth_var = tk.BooleanVar(value=False)
        self.smooth_window_var = tk.IntVar(value=5)

        self.entries = []
        self.run_images = {}
        self._sort_states = {}
        self._plot_points = []
        self._plot_tooltip_ids = []

        self._build_ui()
        self.refresh_data()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text="Worker logs:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.worker_logs_var, width=80).grid(
            row=0, column=1, sticky="we", padx=4
        )
        ttk.Label(top, text="Run logs:").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.run_logs_var, width=80).grid(
            row=1, column=1, sticky="we", padx=4
        )
        ttk.Label(top, text="CSV out:").grid(row=2, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.out_csv_var, width=80).grid(
            row=2, column=1, sticky="we", padx=4
        )

        ttk.Button(top, text="Refresh", command=self.refresh_data).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Export CSV", command=self.export_csv).grid(row=1, column=2, padx=4)

        mode_frame = ttk.Frame(self)
        mode_frame.pack(fill="x", padx=8, pady=4)
        ttk.Label(mode_frame, text="Mode:").pack(side="left")
        ttk.Radiobutton(
            mode_frame, text="Per run", variable=self.mode_var, value="per_run", command=self.update_table
        ).pack(side="left", padx=6)
        ttk.Radiobutton(
            mode_frame, text="All selected runs", variable=self.mode_var, value="overall", command=self.update_table
        ).pack(side="left", padx=6)
        ttk.Label(mode_frame, text="Last N tasks (0=all):").pack(side="left", padx=6)
        last_spin = ttk.Spinbox(
            mode_frame,
            from_=0,
            to=10000,
            textvariable=self.last_n_var,
            width=6,
            command=self.update_table,
        )
        last_spin.pack(side="left", padx=4)
        last_spin.bind("<Return>", lambda _event: self.update_table())
        last_spin.bind("<FocusOut>", lambda _event: self.update_table())

        filter_frame = ttk.Frame(self)
        filter_frame.pack(fill="x", padx=8, pady=4)

        self.run_list = self._make_listbox(filter_frame, "Runs")
        self.host_list = self._make_listbox(filter_frame, "Hosts")
        self.cache_list = self._make_listbox(filter_frame, "Cache")
        self.workers_list = self._make_listbox(filter_frame, "Workers")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=6)

        table_frame = ttk.Frame(notebook)
        plot_frame = ttk.Frame(notebook)
        notebook.add(table_frame, text="Summary")
        notebook.add(plot_frame, text="Plot")

        cols = [
            "run",
            "host",
            "workers",
            "cache",
            "n_ok/total",
            "avg_runtime_s",
            "tput_per_worker_h",
            "tput_total_h",
            "img_tput_per_worker_h",
            "img_tput_total_h",
            "rss_max_mb",
            "rss_p50_mb",
            "rss_p95_mb",
        ]
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        for col in cols:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by(c))
            self.tree.column(col, width=90, anchor="center")
        self.tree.pack(fill="both", expand=True)

        plot_controls = ttk.Frame(plot_frame)
        plot_controls.pack(fill="x", padx=6, pady=4)

        ttk.Label(plot_controls, text="Metric:").pack(side="left")
        self.metric_var = tk.StringVar(value="tput_total_h")
        metric_values = [
            "tput_total_h",
            "tput_per_worker_h",
            "img_tput_total_h",
            "avg_runtime_s",
            "rss_mb",
        ]
        metric_cb = ttk.Combobox(
            plot_controls, textvariable=self.metric_var, values=metric_values, width=20, state="readonly"
        )
        metric_cb.pack(side="left", padx=6)

        ttk.Label(plot_controls, text="Group by:").pack(side="left")
        self.group_var = tk.StringVar(value="host")
        group_values = ["host", "run", "host+run", "all"]
        group_cb = ttk.Combobox(
            plot_controls, textvariable=self.group_var, values=group_values, width=12, state="readonly"
        )
        group_cb.pack(side="left", padx=6)

        ttk.Button(plot_controls, text="Plot", command=self.update_plot).pack(side="left", padx=6)

        ttk.Checkbutton(
            plot_controls, text="Smooth", variable=self.smooth_var, command=self.update_plot
        ).pack(side="left", padx=6)
        ttk.Label(plot_controls, text="Window:").pack(side="left")
        smooth_spin = ttk.Spinbox(
            plot_controls, from_=2, to=50, textvariable=self.smooth_window_var, width=5, command=self.update_plot
        )
        smooth_spin.pack(side="left", padx=4)

        self.plot_canvas = tk.Canvas(plot_frame, background="white")
        self.plot_canvas.pack(fill="both", expand=True, padx=6, pady=6)
        self.plot_canvas.bind("<Motion>", self.on_plot_hover)
        self.plot_canvas.bind("<Leave>", self.on_plot_leave)

    def _make_listbox(self, parent, label):
        frame = ttk.Frame(parent)
        frame.pack(side="left", padx=10)
        ttk.Label(frame, text=label).pack()
        lb = tk.Listbox(frame, selectmode="extended", height=8, exportselection=False)
        lb.pack()
        lb.bind("<<ListboxSelect>>", lambda _event: self.update_table())
        return lb

    def refresh_data(self):
        worker_logs_dir = Path(self.worker_logs_var.get())
        run_logs_dir = Path(self.run_logs_var.get())
        if not worker_logs_dir.exists():
            messagebox.showerror("Error", f"Worker logs path not found: {worker_logs_dir}")
            return
        if not run_logs_dir.exists():
            messagebox.showerror("Error", f"Run logs path not found: {run_logs_dir}")
            return

        self.entries = parse_worker_logs(worker_logs_dir)
        self.run_images = load_run_images(run_logs_dir)
        self._populate_filters()
        self.update_table()

    def _populate_filters(self):
        runs = sorted({e["run"] for e in self.entries})
        hosts = sorted({e["host"] for e in self.entries})
        caches = sorted({e["cache"] for e in self.entries if e["cache"] is not None})
        workers = sorted({e["workers"] for e in self.entries if e["workers"] is not None})

        self._set_listbox(self.run_list, runs)
        self._set_listbox(self.host_list, hosts)
        self._set_listbox(self.cache_list, caches)
        self._set_listbox(self.workers_list, workers)

    def _set_listbox(self, lb, items):
        lb.delete(0, tk.END)
        for item in items:
            lb.insert(tk.END, item)
        for i in range(len(items)):
            lb.selection_set(i)

    def _selected_items(self, lb):
        selected = [lb.get(i) for i in lb.curselection()]
        return selected

    def update_table(self):
        mode = self.mode_var.get()
        runs = self._selected_items(self.run_list)
        hosts = self._selected_items(self.host_list)
        caches = self._selected_items(self.cache_list)
        caches = [int(c) for c in caches] if caches else []
        workers = self._selected_items(self.workers_list)
        workers = [int(w) for w in workers] if workers else []

        try:
            last_n = int(self.last_n_var.get())
        except Exception:
            last_n = 0
        if last_n < 0:
            last_n = 0
        rows = compute_rows(
            self.entries, self.run_images, mode, runs, hosts, caches, workers, last_n
        )

        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in rows:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    row["run"],
                    row["host"],
                    row["workers"],
                    row["cache"],
                    row["ok_total"],
                    f"{row['avg_runtime_s']:.2f}",
                    f"{row['tput_per_worker_h']:.2f}",
                    f"{row['tput_total_h']:.2f}",
                    f"{row['img_tput_per_worker_h']:.2f}",
                    f"{row['img_tput_total_h']:.2f}",
                    f"{row['rss_max_mb']:.0f}",
                    f"{row['rss_p50_mb']:.0f}",
                    f"{row['rss_p95_mb']:.0f}",
                ),
            )
        self.update_plot()

    def export_csv(self):
        path = Path(self.out_csv_var.get())
        rows = [self.tree.item(i)["values"] for i in self.tree.get_children()]
        if not rows:
            messagebox.showinfo("Export", "No rows to export.")
            return
        header = [
            "run",
            "host",
            "workers",
            "cache",
            "n_ok/total",
            "avg_runtime_s",
            "tput_per_worker_h",
            "tput_total_h",
            "img_tput_per_worker_h",
            "img_tput_total_h",
            "rss_max_mb",
            "rss_p50_mb",
            "rss_p95_mb",
        ]
        try:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Export", f"Wrote {len(rows)} rows to {path}")

    def sort_by(self, col):
        numeric_cols = {
            "workers",
            "cache",
            "avg_runtime_s",
            "tput_per_worker_h",
            "tput_total_h",
            "img_tput_per_worker_h",
            "img_tput_total_h",
            "rss_max_mb",
            "rss_p50_mb",
            "rss_p95_mb",
        }
        items = [(self.tree.set(k, col), k) for k in self.tree.get_children("")]
        descending = self._sort_states.get(col, False)

        def parse_num(val):
            if val in ("", "?", None):
                return float("-inf") if descending else float("inf")
            try:
                return float(val)
            except ValueError:
                return float("-inf") if descending else float("inf")

        def parse_ok_total(val):
            try:
                ok, total = val.split("/")
                ok_v = float(ok)
                total_v = float(total)
                return (ok_v / total_v) if total_v > 0 else 0.0
            except Exception:
                return float("-inf") if descending else float("inf")

        if col == "n_ok/total":
            items.sort(key=lambda t: parse_ok_total(t[0]), reverse=descending)
        elif col in numeric_cols:
            items.sort(key=lambda t: parse_num(t[0]), reverse=descending)
        else:
            items.sort(key=lambda t: (t[0] or ""), reverse=descending)

        for index, (_val, k) in enumerate(items):
            self.tree.move(k, "", index)
        self._sort_states[col] = not descending

    def update_plot(self):
        if not self.plot_canvas.winfo_exists():
            return
        width = max(self.plot_canvas.winfo_width(), 800)
        height = max(self.plot_canvas.winfo_height(), 400)
        self.plot_canvas.delete("all")
        self._plot_points = []
        self._clear_tooltip()

        runs = self._selected_items(self.run_list)
        hosts = self._selected_items(self.host_list)
        caches = self._selected_items(self.cache_list)
        caches = [int(c) for c in caches] if caches else []
        workers = self._selected_items(self.workers_list)
        workers = [int(w) for w in workers] if workers else []
        try:
            last_n = int(self.last_n_var.get())
        except Exception:
            last_n = 0
        if last_n < 0:
            last_n = 0
        smooth = self.smooth_var.get()
        smooth_window = max(2, int(self.smooth_window_var.get() or 2))

        group_map = {
            "host": "host",
            "run": "run",
            "host+run": "host_run",
            "all": "all",
        }
        group_by = group_map.get(self.group_var.get(), "host")
        metric = self.metric_var.get()

        series = compute_plot_series(
            self.entries,
            self.run_images,
            group_by,
            runs,
            hosts,
            caches,
            workers,
            metric,
            last_n,
        )

        if not series:
            self.plot_canvas.create_text(
                width // 2, height // 2, text="No data for plot", fill="gray"
            )
            return

        max_len = max(len(v) for v in series.values()) or 1
        max_val = max((max(v) if v else 0.0) for v in series.values()) or 1.0

        pad_left = 50
        pad_right = 20
        pad_top = 20
        pad_bottom = 40
        plot_w = width - pad_left - pad_right
        plot_h = height - pad_top - pad_bottom

        self.plot_canvas.create_line(pad_left, pad_top, pad_left, pad_top + plot_h, fill="#333")
        self.plot_canvas.create_line(
            pad_left, pad_top + plot_h, pad_left + plot_w, pad_top + plot_h, fill="#333"
        )
        # Y-axis ticks
        tick_count = 5
        for i in range(tick_count + 1):
            frac = i / tick_count
            y = pad_top + plot_h - frac * plot_h
            val = max_val * frac
            self.plot_canvas.create_line(pad_left - 4, y, pad_left, y, fill="#333")
            self.plot_canvas.create_text(
                pad_left - 8, y, text=f"{val:.1f}", anchor="e", fill="#333"
            )
        # X-axis ticks
        if max_len > 1:
            x_ticks = min(5, max_len - 1)
            for i in range(x_ticks + 1):
                frac = i / x_ticks
                x = pad_left + frac * plot_w
                idx = int(round(frac * (max_len - 1)))
                self.plot_canvas.create_line(x, pad_top + plot_h, x, pad_top + plot_h + 4, fill="#333")
                self.plot_canvas.create_text(
                    x, pad_top + plot_h + 12, text=str(idx), anchor="n", fill="#333"
                )
        self.plot_canvas.create_text(
            pad_left + plot_w // 2,
            pad_top + plot_h + 25,
            text="task order",
            fill="#333",
        )
        self.plot_canvas.create_text(
            pad_left - 30,
            pad_top + plot_h // 2,
            text=metric,
            angle=90,
            fill="#333",
        )

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        for idx, (key, values) in enumerate(sorted(series.items(), key=lambda kv: str(kv[0]))):
            if not values:
                continue
            color = colors[idx % len(colors)]
            line_values = smooth_series(values, smooth_window) if smooth else values
            points = []
            for i, val in enumerate(values):
                x = pad_left + (i / max(1, max_len - 1)) * plot_w
                y = pad_top + plot_h - (val / max_val) * plot_h
                points.append((x, y))
                self.plot_canvas.create_oval(
                    x - 2, y - 2, x + 2, y + 2, fill=color, outline=""
                )
                label = str(key)
                self._plot_points.append(
                    {
                        "x": x,
                        "y": y,
                        "text": f"{label} | i={i} | {metric}={val:.2f}",
                    }
                )
            if len(line_values) > 1:
                line_points = []
                for i, val in enumerate(line_values):
                    x = pad_left + (i / max(1, max_len - 1)) * plot_w
                    y = pad_top + plot_h - (val / max_val) * plot_h
                    line_points.append((x, y))
                self.plot_canvas.create_line(line_points, fill=color)

        legend_x = pad_left + 10
        legend_y = pad_top + 5
        for idx, key in enumerate(sorted(series.keys(), key=lambda k: str(k))):
            color = colors[idx % len(colors)]
            label = f"{key}"
            self.plot_canvas.create_rectangle(
                legend_x, legend_y + idx * 14, legend_x + 10, legend_y + idx * 14 + 10, fill=color, outline=""
            )
            self.plot_canvas.create_text(
                legend_x + 14, legend_y + idx * 14 + 5, text=label, anchor="w", fill="#333"
            )

    def on_plot_leave(self, _event=None):
        self._clear_tooltip()

    def _clear_tooltip(self):
        if not self._plot_tooltip_ids:
            return
        for item_id in self._plot_tooltip_ids:
            self.plot_canvas.delete(item_id)
        self._plot_tooltip_ids = []

    def on_plot_hover(self, event):
        if not self._plot_points:
            self._clear_tooltip()
            return
        x = event.x
        y = event.y
        best = None
        best_d2 = 64
        for point in self._plot_points:
            dx = point["x"] - x
            dy = point["y"] - y
            d2 = dx * dx + dy * dy
            if d2 <= best_d2:
                best_d2 = d2
                best = point
        if best is None:
            self._clear_tooltip()
            return

        self._clear_tooltip()
        tx = best["x"] + 10
        ty = best["y"] - 10
        text_id = self.plot_canvas.create_text(
            tx, ty, text=best["text"], anchor="nw", fill="#000"
        )
        bbox = self.plot_canvas.bbox(text_id)
        if bbox:
            pad = 4
            rect_id = self.plot_canvas.create_rectangle(
                bbox[0] - pad,
                bbox[1] - pad,
                bbox[2] + pad,
                bbox[3] + pad,
                fill="#ffffe0",
                outline="#333",
            )
            self.plot_canvas.tag_raise(text_id, rect_id)
            self._plot_tooltip_ids = [rect_id, text_id]
        else:
            self._plot_tooltip_ids = [text_id]


if __name__ == "__main__":
    app = StatsUI()
    app.mainloop()
