#!/usr/bin/env python3
"""
explorer.py — Results Explorer UI

Step 1. Add a simple window with a toolbar that includes a filtered combobox
for experiment names. We source experiment names from the *registered configs*
(via configs.list_experiments), and we import `experiments` to ensure all
modules register themselves on import.

Next steps (future edits):
- Wire up "Load" to validate the experiment and show basic info.
- Add a Run ID input and a details panel.
- Add filesystem discovery (optional) and plots preview.
"""

from __future__ import annotations

import os
import sys
import platform
import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

# --- Experiment registry access ------------------------------------------------
# Importing the package triggers registration of experiments via decorators.
# The registry is then exposed by configs.list_experiments().
import experiments as _  # noqa: F401 (import side-effects register experiments)
from configs import list_experiments

APP_TITLE = "Results Explorer"
DEFAULT_GEOMETRY = "900x600"
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", os.path.join(os.getcwd(), "results"))


def _configure_windows_dpi_awareness() -> None:
    """Make the app look crisp on Windows (no-op elsewhere)."""
    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
        except Exception:
            pass


class ResultsExplorerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry(DEFAULT_GEOMETRY)
        self.minsize(720, 420)

        self.results_root = RESULTS_ROOT
        self._all_experiments: List[str] = []

        self._build_menu()
        self._build_layout()
        self._apply_style()

        # Populate experiments after UI is built (so we can update widgets)
        self._refresh_experiments()

    # --- UI construction -----------------------------------------------------
    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Exit", command=self._on_exit)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def _build_layout(self) -> None:
        # Root container
        root = ttk.Frame(self, padding=(10, 10, 10, 6))
        root.pack(fill="both", expand=True)

        # Toolbar panel -------------------------------------------------------
        toolbar_panel = ttk.LabelFrame(root, text="Experiment Selection", padding=(8, 8))
        toolbar_panel.pack(fill="x", pady=(0, 8))

        # Experiment combobox + Load/Refresh
        ttk.Label(toolbar_panel, text="Experiment:").pack(side="left", padx=(0, 8))

        self.experiment_var = tk.StringVar()
        self.experiment_combo = ttk.Combobox(
            toolbar_panel,
            textvariable=self.experiment_var,
            values=[],
            width=40,
            state="normal",  # allow typing
        )
        self.experiment_combo.pack(side="left", padx=(0, 8))
        self.experiment_combo.bind("<KeyRelease>", self._on_experiment_typed)
        self.experiment_combo.bind("<<ComboboxSelected>>", self._on_experiment_selected)
        self.experiment_combo.bind("<Return>", self._on_experiment_selected)

        ttk.Button(toolbar_panel, text="Load", command=self._load_experiment).pack(side="left", padx=(0, 8))
        ttk.Button(toolbar_panel, text="Refresh", command=self._refresh_experiments).pack(side="left")

        # Overview panel ------------------------------------------------------
        overview_panel = ttk.LabelFrame(root, text="Experiment Overview", padding=(8, 8))
        overview_panel.pack(fill="x", pady=(0, 8))

        # Description row
        desc_frame = ttk.Frame(overview_panel)
        desc_frame.pack(fill="x", pady=(0, 4))
        ttk.Label(desc_frame, text="Description:").pack(side="left", padx=(0, 8))
        self.description_var = tk.StringVar(value="No experiment loaded")
        self.description_label = ttk.Label(desc_frame, textvariable=self.description_var, foreground="#333")
        self.description_label.pack(side="left", fill="x", expand=True)

        # Stats row
        stats_frame = ttk.Frame(overview_panel)
        stats_frame.pack(fill="x")
        
        ttk.Label(stats_frame, text="Runs:").pack(side="left", padx=(0, 4))
        self.run_count_var = tk.StringVar(value="—")
        ttk.Label(stats_frame, textvariable=self.run_count_var, foreground="#333").pack(side="left", padx=(0, 16))
        
        ttk.Label(stats_frame, text="Last Updated:").pack(side="left", padx=(0, 4))
        self.last_updated_var = tk.StringVar(value="—")
        ttk.Label(stats_frame, textvariable=self.last_updated_var, foreground="#333").pack(side="left")

        # Placeholder main content area --------------------------------------
        self.title_label = ttk.Label(root, text="Results Explorer", font=("Segoe UI", 16))
        self.title_label.pack(anchor="w")

        self.subtitle_label = ttk.Label(
            root,
            text=f"results root: {self.results_root}",
            foreground="#666",
        )
        self.subtitle_label.pack(anchor="w")

        # Stretchable spacer
        spacer = ttk.Frame(root)
        spacer.pack(fill="both", expand=True)

        # Status bar ----------------------------------------------------------
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(8, 2))
        status.pack(side="bottom", fill="x")

    def _apply_style(self) -> None:
        style = ttk.Style(self)
        try:
            if platform.system() == "Windows":
                style.theme_use("vista")
            elif platform.system() == "Darwin":
                style.theme_use("aqua")
            else:
                style.theme_use("clam")
        except tk.TclError:
            pass

    # --- Experiment data loading --------------------------------------------
    def _load_experiment_overview(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load experiment overview data from analysis_data.json and stats.json."""
        experiment_dir = Path(self.results_root) / experiment_name
        
        analysis_file = experiment_dir / "analysis_data.json"
        stats_file = experiment_dir / "stats.json"
        
        data = {}
        
        # Try to load description from analysis_data.json
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
                # Look for description in basic_stats.experiment_info first, then fall back to top level
                if 'basic_stats' in analysis_data and 'experiment_info' in analysis_data['basic_stats']:
                    data['description'] = analysis_data['basic_stats']['experiment_info']['description']
                else:
                    data['description'] = analysis_data['description']
        except FileNotFoundError:
            data['description'] = 'N/A'
            if stats_file.exists():
                self.status_var.set("Missing analysis_data.json")
        except json.JSONDecodeError as e:
            print(f"ERROR: Malformed JSON in {analysis_file}: {e}")
            sys.exit(1)
        except KeyError as e:
            print(f"ERROR: Missing expected field in {analysis_file}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to read {analysis_file}: {e}")
            sys.exit(1)
            
        # Try to load stats from stats.json
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats_data = json.load(f)
                data['num_runs'] = stats_data['total_runs']  # Direct access, will fail if missing
                
                # Convert timestamp - check if it's already a string or needs conversion
                timestamp = stats_data['timestamp']
                if isinstance(timestamp, str):
                    # Already formatted, use as-is
                    data['last_updated'] = timestamp
                elif isinstance(timestamp, (int, float)):
                    # Unix timestamp, convert it
                    dt = datetime.fromtimestamp(timestamp)
                    data['last_updated'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    print(f"ERROR: Unexpected timestamp format in {stats_file}: {type(timestamp)}")
                    sys.exit(1)
        except FileNotFoundError:
            data['num_runs'] = 'N/A'
            data['last_updated'] = 'N/A'
            if analysis_file.exists():
                self.status_var.set("Missing stats.json")
        except json.JSONDecodeError as e:
            print(f"ERROR: Malformed JSON in {stats_file}: {e}")
            sys.exit(1)
        except KeyError as e:
            print(f"ERROR: Missing expected field in {stats_file}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to read {stats_file}: {e}")
            sys.exit(1)
            
        # Check if both files are missing
        if not analysis_file.exists() and not stats_file.exists():
            data['description'] = 'No experiment data found'
            data['num_runs'] = 'N/A'
            data['last_updated'] = 'N/A'
            
        return data
        
    def _update_overview_panel(self, data: Optional[Dict[str, Any]]) -> None:
        """Update the overview panel with experiment data."""
        if data is None:
            self.description_var.set("No experiment loaded")
            self.run_count_var.set("—")
            self.last_updated_var.set("—")
        else:
            self.description_var.set(data.get('description', 'N/A'))
            self.run_count_var.set(str(data.get('num_runs', 'N/A')))
            self.last_updated_var.set(data.get('last_updated', 'N/A'))

    # --- Experiment list handling -------------------------------------------
    def _refresh_experiments(self) -> None:
        """Populate from the *registered* experiments (authoritative list)."""
        try:
            names = list_experiments()  # Sorted by the helper
            self._all_experiments = list(names)
            self._set_combo_values(self._all_experiments)
            self.status_var.set(f"Loaded {len(self._all_experiments)} experiments from registry.")
        except Exception as e:
            self._all_experiments = []
            self._set_combo_values([])
            self.status_var.set(f"Failed to load experiments: {e}")

    def _set_combo_values(self, values: List[str]) -> None:
        current = self.experiment_var.get()
        self.experiment_combo.configure(values=values)
        # Keep typed text intact; do not auto-overwrite
        if current and current not in values:
            # leave text, but we might optionally open dropdown later
            pass

    def _on_experiment_typed(self, event=None) -> None:
        typed = (self.experiment_var.get() or "").strip()
        if not typed:
            self._set_combo_values(self._all_experiments)
            return
        filtered = [n for n in self._all_experiments if typed.lower() in n.lower()]
        self._set_combo_values(filtered)

    def _on_experiment_selected(self, event=None) -> None:
        name = (self.experiment_var.get() or "").strip()
        if not name:
            self.status_var.set("No experiment selected.")
            return
        self.status_var.set(f"Selected experiment: {name}")

    def _load_experiment(self) -> None:
        name = (self.experiment_var.get() or "").strip()
        if not name:
            self.status_var.set("Enter or select an experiment name.")
            self.experiment_combo.focus_set()
            return
            
        # Load and display overview data
        overview_data = self._load_experiment_overview(name)
        self._update_overview_panel(overview_data)
        
        self.status_var.set(f"Loaded experiment: {name}")

    # --- Actions -------------------------------------------------------------
    def _on_exit(self) -> None:
        self.destroy()

    def _show_about(self) -> None:
        from tkinter import messagebox
        messagebox.showinfo(
            "About",
            f"{APP_TITLE}\n\n"
            "A simple UI to browse experiment results and runs.\n"
            "We'll build the features step-by-step.",
        )


def main(argv: list[str] | None = None) -> int:
    _configure_windows_dpi_awareness()
    app = ResultsExplorerApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())