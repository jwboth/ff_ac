r"""
Queue Launcher  -  GUI for a starte master / watchdog / worker (m.m.) for de
distribuerte koene, uten a huske flagg-syntaks.

Kjor med VS Code sin play-knapp (F5). Krever bare tkinter (innebygd) - importerer
IKKE darsia, bare ast-parser ko-skriptene.

  - Velg Skript + Kommando i nedtrekkene.
  - Skjemaet under viser ALLE flagg med riktig widget (avkrysning for av/pa,
    glidebryter for skala, spinn-boks for tall, nedtrekk for valg, Bla-knapp for
    stier) og en forklaring per felt.
  - "Standard 10 reps" fyller --runs. Profiler lar deg lagre/laste hele oppsett.
  - "Kopier kommando" -> utklippstavle. "Kjor" -> eget konsollvindu.
  - Kommandoen tar kun med felt du har endret fra default (ryddig CLI).
"""
import ast
import glob
import json
import os
import re
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(HERE, "scripts")
STATE = os.path.join(HERE, ".queue_launcher_state.json")
_VARIANT = re.compile(r"(backup|_old\b|_new\b|copy|tmp|temp|\.bak|~|_v\d+|_\d+\b|\(\d+\)|test)", re.I)

STD_RUNS = "ac31 ac26 ac42 ac22 ac27 ac48 ac51 ac50 ac53 ac58"

HELP = {
    "--queue": "Shared queue directory (same path for the master and every watchdog). The master writes tasks here; workers pick them up.",
    "--runs": "Experiments to calibrate, space-separated. Use 'Standard 10 reps' to fill the fleet.",
    "--config-dir": "Folder with <run>.toml configs (e.g. config_seg6/run_ac).",
    "--logs-dir": "Folder for logs and result CSVs.",
    "--ref-config": "Optional global reference config that overrides shared settings.",
    "--use-facies": "Use a separate signal model per geological layer (facies). Recommended ON for AC.",
    "--per-label": "Give each facies its own signal parameters (otherwise shared). Recommended ON.",
    "--use-label-weights": "Weight the objective differently per facies (requires --label-weights).",
    "--auto-label-weights": "Compute label weights automatically when none are given.",
    "--label-weights": "Manual per-facies weights, e.g. '7=1.0;8=0.5'. Requires --use-label-weights.",
    "--label-weight-grouping": "Normalize label weights by group (e.g. per facies) when facies is off.",
    "--enforce-lower": "Penalize solutions where detected mass EXCEEDS injected (force under-estimation).",
    "--objective-integral": "Mass-conservation penalty: adds lambda * total-variation of detected mass over "
                            "the post-injection plateau (closed cell => true mass constant). Counters the "
                            "BTB area-tracking over-detection. lambda=1.0 is well-scaled. (l1/l2 legacy "
                            "values can still be pasted via 'Lim inn kommando'.)",
    "--optuna-persist": "Persist the Optuna study to sqlite so it can be resumed.",
    "--optuna-storage-dir": "Folder for Optuna sqlite databases (defaults to --logs-dir).",
    "--use-last-best": "Seed Optuna with the previous best result instead of warmup.",
    "--use-history": "Seed Optuna with all valid past trials from the history CSV.",
    "--bounds-file": "JSON with parameter bounds (e.g. config/bounds_seg6_coupled.json).",
    "--max-iters": "Number of Optuna iterations per experiment (warmup is extra). seg6 typically converges by ~600-800.",
    "--warmup-iters": "Number of random warmup evaluations per experiment before Optuna starts.",
    "--warmup-levels": "Explicit warmup levels, comma-separated (e.g. 1.0,0.75,0.5).",
    "--param-ranges": "Override per-value bounds. Format: 'value2=0,1;value6=0,2'.",
    "--param-levels": "Levels in the structured warmup grid ('8' or 'value2=8;value6=6').",
    "--warmup-high": "Cap for high warmup values (signal.label*.value1..N).",
    "--warmup-mode": "Warmup sweep shape: prefix, suffix or single (legacy).",
    "--skip-warmup": "Skip warmup entirely (useful with --use-last-best and unchanged data).",
    "--run-mode": "parallel = all experiments at once; sequential/serial = one at a time.",
    "--max-in-flight": "Global cap on concurrent tasks across all experiments (0 = off).",
    "--max-in-flight-per-run": "Max concurrent tasks PER experiment (x number of experiments = total in-flight). Controls the pace.",
    "--control-dir": "Folder for live control during a run (max in-flight, worker limits via txt files).",
    "--poll-seconds": "How often the master polls the queue and results (seconds).",
    "--max-retries": "How many times a failed/timed-out task is retried.",
    "--task-timeout-minutes": "A task is considered failed if it does not finish within this.",
    "--heartbeat-timeout-seconds": "A worker is considered dead if its heartbeat is older than this.",
    "--master-mem-log-seconds": "How often master/system memory is logged (seconds).",
    "--sanity-every": "Run a full-scale cross-check on the best every N iterations (0 = off). Use with --quality-scale.",
    "--sanity-scale": "Quality for the periodic cross-check (1.0 = full scale, recommended).",
    "--sanity-dtype": "Dtype for the sanity cross-check (defaults to --quality-dtype).",
    "--memmap-cache": "Shared memmap cache for images/arrays (off/images/arrays/all) - saves memory across workers.",
    "--memmap-dir": "Folder for the shared memmap cache (prefer local disk, not network).",
    "--quality-scale": "Downscale factor for worker evaluation (1.0 = full). Lower = faster, less memory traffic. Master is always 1.0.",
    "--quality-dtype": "Numeric precision for worker evaluation. float32 = ~2x less memory traffic, practically lossless.",
    "--workers": "Number of worker processes this watchdog starts and keeps alive.",
    "--worker-cmd": "Command template for starting a worker (advanced; default is fine).",
    "--worker-stall-seconds": "Kill and restart a worker if its heartbeat is older than this (hung).",
}

COMMON = {
    "--queue", "--runs", "--config-dir", "--bounds-file", "--control-dir", "--logs-dir",
    "--max-iters", "--warmup-iters", "--run-mode", "--max-in-flight-per-run",
    "--use-facies", "--per-label", "--quality-scale", "--quality-dtype",
    "--sanity-every", "--sanity-scale", "--workers", "--worker-stall-seconds",
    "--objective-integral",
}
SPIN = {
    "--max-iters": (0, 10000, 50), "--warmup-iters": (0, 2000, 10),
    "--workers": (1, 128, 1), "--max-retries": (0, 20, 1),
    "--max-in-flight": (0, 400, 1), "--max-in-flight-per-run": (0, 50, 1),
    "--task-timeout-minutes": (1, 600, 5), "--heartbeat-timeout-seconds": (30, 3600, 30),
    "--sanity-every": (0, 5000, 10), "--worker-stall-seconds": (0, 3600, 30),
}
SCALE_FLAGS = {"--quality-scale", "--sanity-scale"}


def is_path(flag):
    return flag.endswith("-dir") or flag.endswith("-file") or flag in ("--queue", "--config-dir", "--ref-config")
def pick_dir(flag):
    return flag.endswith("-dir") or flag in ("--queue", "--config-dir")
def is_dtype(flag):
    return flag.endswith("dtype")
def is_bool(spec):
    return spec["action"] == "store_true" or spec["type"] == "_parse_bool"


def _const(n): return n.value if isinstance(n, ast.Constant) else None
def _name(n): return n.id if isinstance(n, ast.Name) else (n.attr if isinstance(n, ast.Attribute) else None)
def _list(n): return [e.value for e in n.elts if isinstance(e, ast.Constant)] if isinstance(n, ast.List) else None


def parse_add_argument(node):
    flags = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
    if not flags:
        return None
    kw = {k.arg: k.value for k in node.keywords if k.arg}
    lf = next((f for f in flags if f.startswith("--")), flags[0])
    spec = {
        "flags": flags, "label": lf,
        "type": _name(kw["type"]) if "type" in kw else None,
        "default": _const(kw["default"]) if "default" in kw else None,
        "action": _const(kw["action"]) if "action" in kw else None,
        "choices": _list(kw["choices"]) if "choices" in kw else None,
        "nargs": _const(kw["nargs"]) if "nargs" in kw else None,
        "required": bool(_const(kw["required"])) if "required" in kw else False,
        "help": _const(kw["help"]) if "help" in kw else None,
        "positional": not flags[0].startswith("-"),
    }
    spec["help"] = HELP.get(lf) or spec["help"] or ""
    return spec


def extract_specs(path):
    try:
        tree = ast.parse(open(path, encoding="utf-8").read())
    except Exception:
        return {}
    subvars = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            f = n.value.func
            if isinstance(f, ast.Attribute) and f.attr == "add_parser" and n.value.args:
                a0 = n.value.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            subvars[t.id] = a0.value
    specs = {c: [] for c in subvars.values()}
    for n in ast.walk(tree):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "add_argument"
                and isinstance(n.func.value, ast.Name) and n.func.value.id in subvars):
            s = parse_add_argument(n)
            if s:
                specs[subvars[n.func.value.id]].append(s)
    return specs


def discover_scripts(d):
    files = [f for f in sorted(glob.glob(os.path.join(d, "distributed_*queue*.py")))
             if not _VARIANT.search(os.path.basename(f))]
    out = {}
    for f in files:
        st = os.path.basename(f).replace("distributed_", "").replace("_queue", "").replace(".py", "")
        if st not in out or len(os.path.basename(f)) < len(os.path.basename(out[st])):
            out[st] = f
    return out


CMD_ORDER = {"master": 0, "supervisor": 0, "watchdog": 1, "worker": 2}


# ----------------------------- GUI -----------------------------

class App:
    def __init__(self, root, scripts):
        self.root = root
        self.scripts = scripts
        self.specs = {s: extract_specs(p) for s, p in scripts.items()}
        self.vars = {}
        self.state = {}
        if os.path.exists(STATE):
            try:
                self.state = json.load(open(STATE, encoding="utf-8"))
            except Exception:
                self.state = {}
        root.title("Queue Launcher"); root.geometry("980x720")

        top = ttk.Frame(root); top.pack(fill="x", padx=10, pady=(8, 2))
        ttk.Label(top, text="Python:").grid(row=0, column=0, sticky="w")
        self.py_var = tk.StringVar(value=self.state.get("python", sys.executable))
        ttk.Entry(top, textvariable=self.py_var).grid(row=0, column=1, columnspan=6, sticky="we", padx=4)
        top.columnconfigure(1, weight=1)
        stems = sorted(self.scripts, key=lambda s: (0 if "calibr" in s else 1, s))
        ttk.Label(top, text="Skript:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.script_var = tk.StringVar(value=self.state.get("script", stems[0] if stems else ""))
        self.script_cb = ttk.Combobox(top, textvariable=self.script_var, values=stems, state="readonly", width=22)
        self.script_cb.grid(row=1, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(top, text="Kommando:").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.cmd_var = tk.StringVar(value=self.state.get("command", ""))
        self.cmd_cb = ttk.Combobox(top, textvariable=self.cmd_var, state="readonly", width=20)
        self.cmd_cb.grid(row=1, column=3, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(top, text="Profil:").grid(row=1, column=4, sticky="w", pady=(6, 0))
        self.profile_var = tk.StringVar()
        self.profile_cb = ttk.Combobox(top, textvariable=self.profile_var, state="readonly", width=26,
                                       values=self._profile_values())
        self.profile_cb.grid(row=1, column=5, sticky="w", padx=4, pady=(6, 0))
        self.profile_cb.bind("<<ComboboxSelected>>", lambda e: self.load_profile())
        self.adv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Vis avanserte valg", variable=self.adv_var,
                        command=self._rebuild_form).grid(row=1, column=6, sticky="e", padx=4, pady=(6, 0))
        self.script_cb.bind("<<ComboboxSelected>>", lambda e: self._on_script())
        self.cmd_cb.bind("<<ComboboxSelected>>", lambda e: self._rebuild_form())

        body = ttk.Frame(root); body.pack(fill="both", expand=True, padx=10, pady=4)
        self.canvas = tk.Canvas(body, highlightthickness=0)
        vsb = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y"); self.canvas.pack(side="left", fill="both", expand=True)
        self.form = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.form, anchor="nw")
        self.form.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-e.delta / 120), "units"))

        bar = ttk.Frame(root); bar.pack(fill="x", padx=10, pady=6)
        ttk.Button(bar, text="Kopier kommando", command=self.copy_cmd).pack(side="left")
        ttk.Button(bar, text="Kjor", command=self.run_cmd).pack(side="left", padx=6)
        ttk.Button(bar, text="Lim inn kommando", command=self.paste_command).pack(side="left", padx=(0, 12))
        ttk.Button(bar, text="Lagre som ny...", command=self.save_profile).pack(side="left")
        ttk.Button(bar, text="Oppdater valgt", command=self.update_profile).pack(side="left", padx=6)
        ttk.Button(bar, text="Slett profil", command=self.delete_profile).pack(side="left")
        self.preview = tk.StringVar(value="")
        ttk.Label(root, textvariable=self.preview, foreground="#06c", wraplength=940,
                  justify="left").pack(fill="x", padx=10, pady=(0, 8))
        self._on_script(initial=True)

    def _commands(self, stem):
        return sorted(self.specs.get(stem, {}), key=lambda c: (CMD_ORDER.get(c, 9), c))

    def _on_script(self, initial=False):
        stem = self.script_var.get(); cmds = self._commands(stem)
        self.cmd_cb["values"] = cmds
        if not (initial and self.cmd_var.get() in cmds):
            self.cmd_var.set(cmds[0] if cmds else "")
        self._rebuild_form()

    def _var(self, stem, cmd, spec):
        key = (stem, cmd, spec["label"])
        if key not in self.vars:
            saved = self.state.get("values", {}).get(stem, {}).get(cmd, {}).get(spec["label"])
            if is_bool(spec):
                self.vars[key] = tk.BooleanVar(value=bool(saved) if saved is not None else bool(spec["default"]))
            elif spec["label"] in SCALE_FLAGS:
                d = saved if saved not in (None, "") else (spec["default"] if spec["default"] not in (None, "") else 1.0)
                self.vars[key] = tk.DoubleVar(value=float(d))
            else:
                d = "" if spec["default"] is None else str(spec["default"])
                self.vars[key] = tk.StringVar(value=str(saved) if saved is not None else d)
        return self.vars[key]

    def _rebuild_form(self):
        for w in self.form.winfo_children():
            w.destroy()
        self._aux = []  # keep refs to helper tk-Variables so they are not garbage-collected
        stem, cmd = self.script_var.get(), self.cmd_var.get()
        ttk.Label(self.form, text="-> " + os.path.basename(self.scripts.get(stem, "")) + "   " + cmd,
                  foreground="#888").grid(row=0, column=0, columnspan=4, sticky="w", padx=6, pady=(4, 4))
        specs = self.specs.get(stem, {}).get(cmd, [])
        if not self.adv_var.get():
            common = [s for s in specs if s["label"] in COMMON or s["required"]]
            specs = common if len(common) >= 3 else specs
        r = 1
        for spec in specs:
            flag = spec["label"]; var = self._var(stem, cmd, spec)
            ttk.Label(self.form, text=flag + ("  *" if spec["required"] else ""),
                      font=("Segoe UI", 9, "bold")).grid(row=r, column=0, sticky="nw", padx=(6, 8), pady=(5, 0))
            cell = ttk.Frame(self.form); cell.grid(row=r, column=1, sticky="w", pady=(5, 0))
            if is_bool(spec):
                ttk.Checkbutton(cell, variable=var).pack(side="left")
            elif flag in SCALE_FLAGS:
                lbl = ttk.Label(cell, width=5)
                sc = ttk.Scale(cell, from_=0.25, to=1.0, variable=var, length=180,
                               command=lambda v, L=lbl: L.config(text=f"{float(v):.2f}"))
                sc.pack(side="left"); lbl.config(text=f"{var.get():.2f}"); lbl.pack(side="left", padx=6)
            elif is_dtype(flag):
                ttk.Combobox(cell, textvariable=var, values=["", "float32", "float64"],
                             state="readonly", width=12).pack(side="left")
            elif flag == "--objective-integral":
                # Checkbox + lambda-felt. StringVar-en (var) er sannheten og holder
                # "off" eller "drift[:LAMBDA]" - profiler/lim-inn virker som for.
                cur = str(var.get() or "off").strip().lower()
                on = tk.BooleanVar(value=cur.startswith("drift"))
                lam_init = cur.split(":", 1)[1] if (cur.startswith("drift") and ":" in cur) else "1.0"
                lam = tk.StringVar(value=lam_init)
                self._aux += [on, lam]
                def _sync_drift(v=var, o=on, L=lam):
                    if o.get():
                        s = L.get().strip()
                        v.set("drift" if s in ("", "1", "1.0") else "drift:" + s)
                    else:
                        v.set("off")
                ttk.Checkbutton(cell, text="drift (konserveringsstraff)", variable=on,
                                command=_sync_drift).pack(side="left")
                ttk.Label(cell, text="lambda:").pack(side="left", padx=(10, 2))
                ttk.Entry(cell, textvariable=lam, width=6).pack(side="left")
                lam.trace_add("write", lambda *_a, f=_sync_drift: f())
            elif spec["choices"]:
                ttk.Combobox(cell, textvariable=var, values=[str(c) for c in spec["choices"]],
                             state="readonly", width=20).pack(side="left")
            elif flag in SPIN:
                lo, hi, st = SPIN[flag]
                ttk.Spinbox(cell, from_=lo, to=hi, increment=st, textvariable=var, width=12).pack(side="left")
            else:
                ttk.Entry(cell, textvariable=var, width=40).pack(side="left")
                if is_path(flag):
                    ttk.Button(cell, text="Bla...", width=6,
                               command=lambda v=var, f=flag: self._browse(v, f)).pack(side="left", padx=4)
                if flag == "--runs":
                    ttk.Button(cell, text="Standard 10 reps", command=lambda v=var: v.set(STD_RUNS)).pack(side="left", padx=4)
            meta = []
            if spec["type"] and spec["type"] != "_parse_bool":
                meta.append(spec["type"])
            if spec["nargs"] in ("+", "*"):
                meta.append("multiple, space-separated")
            mt = ("  [" + ", ".join(meta) + "]") if meta else ""
            ttk.Label(self.form, text=(spec["help"] or "(no description)") + mt, foreground="#555",
                      wraplength=540, justify="left").grid(row=r, column=2, sticky="w", padx=8, pady=(5, 0))
            r += 1
        self.form.columnconfigure(2, weight=1)

    def _browse(self, var, flag):
        p = filedialog.askdirectory() if pick_dir(flag) else filedialog.askopenfilename()
        if p:
            var.set(p)

    def build_argv(self):
        stem, cmd = self.script_var.get(), self.cmd_var.get()
        argv = [self.py_var.get(), self.scripts.get(stem, ""), cmd]; pos = []
        for spec in self.specs.get(stem, {}).get(cmd, []):
            flag = spec["label"]; var = self._var(stem, cmd, spec)
            if spec["action"] == "store_true":
                if bool(var.get()):
                    argv.append(flag)
                continue
            if spec["type"] == "_parse_bool":
                if bool(var.get()) != bool(spec["default"]):
                    argv += [flag, "true" if var.get() else "false"]
                continue
            if flag in SCALE_FLAGS:
                val = f"{float(var.get()):.2f}"
                if spec["default"] is None or abs(float(var.get()) - float(spec["default"])) > 1e-9:
                    argv += [flag, val]
                continue
            val = str(var.get()).strip()
            if val == "" or (spec["default"] is not None and val == str(spec["default"]) and not spec["required"]):
                continue
            pieces = val.split() if spec["nargs"] in ("+", "*") else [val]
            if spec["positional"]:
                pos += pieces
            else:
                argv.append(flag); argv += pieces
        return argv[:3] + pos + argv[3:]

    @staticmethod
    def _q(s):
        return f'"{s}"' if (" " in s or "\\" in s) else s
    def _line(self):
        return " ".join(self._q(a) for a in self.build_argv())

    def copy_cmd(self):
        line = self._line(); self.root.clipboard_clear(); self.root.clipboard_append(line)
        self.preview.set("Kopiert:  " + line); self._persist()

    def run_cmd(self):
        argv = self.build_argv(); self.preview.set("Kjorer:  " + self._line()); self._persist()
        # Run from the REPO ROOT (where queue_launcher.py + scripts/ + config*/ live) so
        # relative paths like --config-dir config_seg6/run_ac resolve correctly.
        try:
            if os.name == "nt":
                # 'cmd /k' keeps the console OPEN after the command exits, so you can
                # read output / errors instead of the window flashing shut.
                cmdline = subprocess.list2cmdline(argv)
                subprocess.Popen('cmd /k ' + cmdline, cwd=HERE,
                                 creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(argv, cwd=HERE)
        except Exception as exc:
            messagebox.showerror("Feil ved start", str(exc))

    def _values(self):
        vals = self.state.get("values", {})
        for (stem, cmd, flag), var in self.vars.items():
            vals.setdefault(stem, {}).setdefault(cmd, {})[flag] = var.get()
        return vals

    def _cmd_values(self, stem, cmd):
        """Fresh, independent dict of the current form values for ONE command.
        Profiles must store a COPY (not a reference into the shared state), otherwise
        two profiles of the same command alias the same dict and edit each other."""
        return {flag: var.get() for (s, c, flag), var in self.vars.items()
                if s == stem and c == cmd}

    def _persist(self):
        self.state["python"] = self.py_var.get()
        self.state["script"] = self.script_var.get()
        self.state["command"] = self.cmd_var.get()
        self.state["values"] = self._values()
        try:
            json.dump(self.state, open(STATE, "w", encoding="utf-8"), indent=2)
        except Exception:
            pass

    def save_profile(self):
        stem, cmd = self.script_var.get(), self.cmd_var.get()
        name = simpledialog.askstring("Lagre som ny profil",
                                      f"Navn pa profilen  (lagres som {stem}:{cmd}).\n"
                                      f"Endre navnet for a kopiere den valgte profilen:",
                                      initialvalue=self._profile_name())
        if not name:
            return
        self._persist()
        prof = self.state.setdefault("profiles", {})
        prof[name] = {"python": self.py_var.get(), "script": stem, "command": cmd,
                      "values": {stem: {cmd: self._cmd_values(stem, cmd)}}}
        self.profile_cb["values"] = self._profile_values()
        self.profile_var.set(f"{name}  [{cmd}]"); self._persist()
        self.preview.set(f"Profil '{name}' lagret som {stem}:{cmd}.")

    def _profile_values(self):
        prof = self.state.get("profiles", {})
        return [f"{n}  [{prof[n].get('command', '?')}]" for n in sorted(prof)]

    def _profile_name(self):
        sel = self.profile_var.get()
        return sel.split("  [")[0] if sel else ""

    def load_profile(self):
        name = self._profile_name(); prof = self.state.get("profiles", {}).get(name)
        if not prof:
            return
        self.py_var.set(prof.get("python", self.py_var.get()))
        self.script_var.set(prof.get("script", self.script_var.get()))
        # merge profile values into state so _var picks them up
        for stem, cmds in prof.get("values", {}).items():
            for cmd, flags in cmds.items():
                self.state.setdefault("values", {}).setdefault(stem, {}).setdefault(cmd, {}).update(flags)
        # drop cached vars for this script so they reload from state
        self.vars = {k: v for k, v in self.vars.items() if k[0] != prof.get("script")}
        # set the command list for the profile's script, THEN restore its command.
        self.cmd_cb["values"] = self._commands(self.script_var.get())
        self.cmd_var.set(prof.get("command", ""))
        # initial=True so _on_script keeps the command instead of resetting it to master.
        self._on_script(initial=True)
        self.preview.set(f"Profil '{name}' lastet  ({prof.get('command', '?')}).")

    def update_profile(self):
        """Overwrite the currently selected profile with the current form (edit)."""
        name = self._profile_name(); prof = self.state.get("profiles", {})
        if not name or name not in prof:
            messagebox.showinfo("Oppdater profil", "Velg en profil i nedtrekket forst."); return
        self._persist()
        stem, cmd = self.script_var.get(), self.cmd_var.get()
        prof[name] = {"python": self.py_var.get(), "script": stem, "command": cmd,
                      "values": {stem: {cmd: self._cmd_values(stem, cmd)}}}
        self.profile_cb["values"] = self._profile_values()
        self.profile_var.set(f"{name}  [{cmd}]"); self._persist()
        self.preview.set(f"Profil '{name}' oppdatert ({stem}:{cmd}).")

    def delete_profile(self):
        name = self._profile_name(); prof = self.state.get("profiles", {})
        if name in prof:
            del prof[name]; self.profile_cb["values"] = self._profile_values(); self.profile_var.set("")
            self._persist(); self.preview.set(f"Profil '{name}' slettet.")

    # ---- paste an existing command and fill the form to match ----
    def paste_command(self):
        win = tk.Toplevel(self.root); win.title("Lim inn kommando"); win.geometry("780x240")
        ttk.Label(win, text="Lim inn en full kommando (master/watchdog/worker). "
                            "Linjeskift og PowerShell-fortsettelser (^ / `) fjernes automatisk.",
                  wraplength=750).pack(anchor="w", padx=10, pady=(8, 4))
        txt = tk.Text(win, height=8, wrap="word"); txt.pack(fill="both", expand=True, padx=10)
        def go():
            line = txt.get("1.0", "end"); win.destroy(); self.apply_command(line)
        ttk.Button(win, text="Tolk og fyll inn", command=go).pack(pady=8)
        txt.focus_set()

    @staticmethod
    def _tokenize(line):
        import shlex
        line = line.replace("`", " ").replace("^", " ").replace("\r", " ").replace("\n", " ")
        try:
            toks = shlex.split(line, posix=False)
        except Exception:
            toks = line.split()
        out = []
        for t in toks:
            if t and t != "&":
                if len(t) >= 2 and t[0] in "\"'" and t[-1] == t[0]:
                    t = t[1:-1]
                out.append(t)
        return out

    @staticmethod
    def _is_flag(t):
        return t.startswith("--") or (t.startswith("-") and len(t) > 1 and not (t[1].isdigit() or t[1] == "."))

    def apply_command(self, line):
        toks = self._tokenize(line)
        pyi = next((i for i, t in enumerate(toks) if t.lower().endswith(".py")), None)
        if pyi is None or pyi + 1 >= len(toks):
            messagebox.showwarning("Tolking", "Fant ikke '<skript>.py <kommando>' i kommandoen."); return
        script_base = re.split(r"[\\/]", toks[pyi])[-1]; cmd = toks[pyi + 1]
        stem = next((s for s, p in self.scripts.items() if os.path.basename(p) == script_base), None)
        if stem is None:
            key = script_base.lower().replace("_", "")
            stem = next((s for s in self.scripts if s.replace("_", "") in key), None)
        if stem is None or cmd not in self.specs.get(stem, {}):
            messagebox.showwarning("Tolking", f"Ukjent skript/kommando: {script_base} {cmd}"); return
        specs = self.specs[stem][cmd]
        by_flag = {f: spec for spec in specs for f in spec["flags"]}
        rest = toks[pyi + 2:]
        parsed, unknown, i = {}, [], 0
        while i < len(rest):
            t = rest[i]
            if self._is_flag(t):
                spec = by_flag.get(t)
                if spec and spec["action"] == "store_true":
                    parsed[spec["label"]] = True; i += 1
                else:
                    vals = []; i += 1
                    while i < len(rest) and not self._is_flag(rest[i]):
                        vals.append(rest[i]); i += 1
                    if spec:
                        parsed[spec["label"]] = vals
                    else:
                        unknown.append(t)
            else:
                i += 1
        if pyi > 0 and not self._is_flag(toks[pyi - 1]):
            self.py_var.set(toks[pyi - 1])
        # select script + command, then reset this command's vars to defaults
        self.script_var.set(stem); self.cmd_cb["values"] = self._commands(stem); self.cmd_var.set(cmd)
        self.vars = {k: v for k, v in self.vars.items() if not (k[0] == stem and k[1] == cmd)}
        self.state.get("values", {}).get(stem, {}).pop(cmd, None)
        for spec in specs:
            var = self._var(stem, cmd, spec)                 # default
            if spec["label"] not in parsed:
                continue
            v = parsed[spec["label"]]
            try:
                if is_bool(spec):
                    if spec["action"] == "store_true":
                        var.set(True)
                    else:
                        var.set(str(v[0]).lower() in ("true", "1", "yes", "on") if v else True)
                elif spec["label"] in SCALE_FLAGS:
                    var.set(float(v[0]))
                else:
                    var.set(" ".join(v))
            except Exception:
                pass
        self._rebuild_form()
        msg = f"Tolket: {stem} {cmd}  -  {len(parsed)} flagg satt"
        if unknown:
            msg += f"  (ukjente, hoppet over: {', '.join(unknown)})"
        self.preview.set(msg)


if __name__ == "__main__":
    scripts = discover_scripts(SCRIPTS_DIR)
    if not scripts:
        print("Fant ingen ko-skript i", SCRIPTS_DIR)
    root = tk.Tk()
    App(root, scripts)
    root.mainloop()
