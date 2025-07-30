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
import numpy as np
import ast

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

        # Bind mousewheel scrolling
        self.bind_all("<MouseWheel>", self._on_mousewheel)  # Windows
        self.bind_all("<Button-4>", lambda e: self._on_mousewheel(type('obj', (object,), {'delta': 120})()))  # Linux
        self.bind_all("<Button-5>", lambda e: self._on_mousewheel(type('obj', (object,), {'delta': -120})()))  # Linux

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
        # Create main canvas and scrollbar for entire app
        main_canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        
        # Scrollable frame that will contain all content
        scrollable_frame = ttk.Frame(main_canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Root container (now inside scrollable_frame instead of self)
        root = ttk.Frame(scrollable_frame, padding=(10, 10, 10, 6))
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

        # Accuracy panel ------------------------------------------------------
        accuracy_panel = ttk.LabelFrame(root, text="Accuracy Distribution", padding=(8, 8))
        accuracy_panel.pack(fill="x", pady=(0, 8))

        self.accuracy_widgets = {}
        accuracy_levels = [0, 25, 50, 75, 100]
        
        for level in accuracy_levels:
            level_frame = ttk.Frame(accuracy_panel)
            level_frame.pack(fill="x", pady=1)
            
            # Level label and count
            level_label = ttk.Label(level_frame, text=f"{level}%:", width=6)
            level_label.pack(side="left", padx=(0, 8))
            
            count_var = tk.StringVar(value="—")
            count_label = ttk.Label(level_frame, textvariable=count_var, foreground="#333")
            count_label.pack(side="left", padx=(0, 8))
            
            # Show runs toggle button
            runs_var = tk.BooleanVar()
            runs_button = ttk.Checkbutton(level_frame, text="Show runs", variable=runs_var,
                                        command=lambda l=level: self._toggle_runs_display(l))
            runs_button.pack(side="left", padx=(0, 8))
            
            # Runs display (initially hidden)
            runs_label = ttk.Label(level_frame, text="", foreground="#666", font=("Consolas", 8))
            
            self.accuracy_widgets[level] = {
                'count_var': count_var,
                'runs_var': runs_var,
                'runs_button': runs_button,
                'runs_label': runs_label,
                'runs_data': []
            }

        # Epochs panel --------------------------------------------------------
        epochs_panel = ttk.LabelFrame(root, text="Epoch Percentiles", padding=(8, 8))
        epochs_panel.pack(fill="x", pady=(0, 8))

        # Create grid for percentiles
        percentiles_frame = ttk.Frame(epochs_panel)
        percentiles_frame.pack(fill="x")
        
        # Headers
        ttk.Label(percentiles_frame, text="Percentile", font=("TkDefaultFont", 9, "bold")).grid(row=0, column=0, padx=(0, 16), sticky="w")
        ttk.Label(percentiles_frame, text="Epochs", font=("TkDefaultFont", 9, "bold")).grid(row=0, column=1, sticky="w")
        
        self.epochs_widgets = {}
        percentile_labels = ["0th", "10th", "25th", "50th", "75th", "90th", "100th"]
        
        for i, pct in enumerate(percentile_labels):
            row = i + 1
            ttk.Label(percentiles_frame, text=f"{pct}:").grid(row=row, column=0, padx=(0, 16), sticky="w")
            
            epochs_var = tk.StringVar(value="—")
            epochs_label = ttk.Label(percentiles_frame, textvariable=epochs_var, foreground="#333")
            epochs_label.grid(row=row, column=1, sticky="w")
            
            self.epochs_widgets[pct] = epochs_var

        # Analysis Results panel ----------------------------------------------
        analysis_panel = ttk.LabelFrame(root, text="Analysis Results", padding=(8, 8))
        analysis_panel.pack(fill="both", expand=True, pady=(0, 8))

        # Clustering section
        clustering_frame = ttk.LabelFrame(analysis_panel, text="Clustering Analysis", padding=(6, 6))
        clustering_frame.pack(fill="both", expand=True, pady=(0, 4))
        
        self.clustering_container = ttk.Frame(clustering_frame)
        self.clustering_container.pack(fill="both", expand=True)
        
        self.clustering_status_var = tk.StringVar(value="No clustering data loaded")
        self.clustering_status_label = ttk.Label(self.clustering_container, textvariable=self.clustering_status_var, foreground="#666")
        self.clustering_status_label.pack()

        # Mirror Weights section  
        mirror_frame = ttk.LabelFrame(analysis_panel, text="Mirror Weights Analysis", padding=(6, 6))
        mirror_frame.pack(fill="x", pady=(4, 0))
        
        # Summary row
        mirror_summary_frame = ttk.Frame(mirror_frame)
        mirror_summary_frame.pack(fill="x", pady=(0, 4))
        
        self.mirror_summary_var = tk.StringVar(value="No mirror weights data loaded")
        ttk.Label(mirror_summary_frame, textvariable=self.mirror_summary_var, foreground="#333").pack(side="left")
        
        # Expandable details
        self.mirror_details_var = tk.BooleanVar()
        self.mirror_details_button = ttk.Checkbutton(mirror_summary_frame, text="Show details", 
                                                    variable=self.mirror_details_var,
                                                    command=self._toggle_mirror_details)
        self.mirror_details_button.pack(side="right")
        
        # Details container (initially hidden)
        self.mirror_details_frame = ttk.Frame(mirror_frame)
        self.mirror_details_text = tk.Text(self.mirror_details_frame, height=8, wrap=tk.WORD, 
                                          font=("Consolas", 8), foreground="#666")
        scrollbar = ttk.Scrollbar(self.mirror_details_frame, orient="vertical", command=self.mirror_details_text.yview)
        self.mirror_details_text.configure(yscrollcommand=scrollbar.set)
        
        self.mirror_details_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

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
            analysis_data = None
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
            stats_data = None
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
            
        # Load accuracy and epochs data for Chunk 2
        data['accuracy'] = self._load_accuracy_data(analysis_data, stats_data)
        data['epochs'] = self._load_epochs_data(analysis_data)
        
        # Load clustering and mirror weights data for Chunk 3
        data['clustering'] = self._load_clustering_data(analysis_data)
        data['mirror_weights'] = self._load_mirror_weights_data(analysis_data)
            
        return data
        
    def _load_accuracy_data(self, analysis_data: Optional[Dict], stats_data: Optional[Dict]) -> Dict[str, Any]:
        """Load accuracy bucket counts and run IDs."""
        accuracy_data = {'counts': {}, 'run_ids': {}}
        
        # Try analysis_data.json first
        if analysis_data:
            try:
                # Primary: accuracy.distribution_analysis.level_counts
                if 'accuracy' in analysis_data and 'distribution_analysis' in analysis_data['accuracy']:
                    level_counts = analysis_data['accuracy']['distribution_analysis']['level_counts']
                    accuracy_data['counts'] = {
                        0: level_counts['0.0'],
                        25: level_counts['0.25'], 
                        50: level_counts['0.5'],
                        75: level_counts['0.75'],
                        100: level_counts['1.0']
                    }
                    
                    # Get run IDs if available
                    if 'basic_stats' in analysis_data and 'distributions' in analysis_data['basic_stats']:
                        acc_dist = analysis_data['basic_stats']['distributions']['accuracy_distribution']
                        if 'run_ids' in acc_dist:
                            run_ids = acc_dist['run_ids']
                            accuracy_data['run_ids'] = {
                                0: run_ids['0.0'],
                                25: run_ids['0.25'],
                                50: run_ids['0.5'], 
                                75: run_ids['0.75'],
                                100: run_ids['1.0']
                            }
                # Fallback: basic_stats.distributions.accuracy_distribution.bins
                elif 'basic_stats' in analysis_data and 'distributions' in analysis_data['basic_stats']:
                    acc_dist = analysis_data['basic_stats']['distributions']['accuracy_distribution']
                    if 'bins' in acc_dist:
                        bins = acc_dist['bins']
                        accuracy_data['counts'] = {
                            0: bins['0.0'],
                            25: bins['0.25'],
                            50: bins['0.5'],
                            75: bins['0.75'],
                            100: bins['1.0']
                        }
                        
                        # Get run IDs if available
                        if 'run_ids' in acc_dist:
                            run_ids = acc_dist['run_ids']
                            accuracy_data['run_ids'] = {
                                0: run_ids['0.0'],
                                25: run_ids['0.25'],
                                50: run_ids['0.5'],
                                75: run_ids['0.75'],
                                100: run_ids['1.0']
                            }
            except KeyError:
                # Fall through to stats_data fallback
                pass
                
        # Fallback to stats.json
        if not accuracy_data['counts'] and stats_data and 'accuracy_distribution' in stats_data:
            acc_dist = stats_data['accuracy_distribution']
            accuracy_data['counts'] = {
                0: acc_dist['0_percent'],
                25: acc_dist['25_percent'],
                50: acc_dist['50_percent'], 
                75: acc_dist['75_percent'],
                100: acc_dist['100_percent']
            }
            # No run IDs in stats.json
            
        return accuracy_data
        
    def _load_epochs_data(self, analysis_data: Optional[Dict]) -> Dict[str, Any]:
        """Load epoch percentiles data."""
        epochs_data = analysis_data["convergence_timing"]["all_runs"]
        return epochs_data
        
    def _load_clustering_data(self, analysis_data: Optional[Dict]) -> Dict[str, Any]:
        """Load clustering analysis data."""
        clustering_data = {'layers': {}}
        
        if not analysis_data:
            return clustering_data
            
        try:
            # Primary: hyperplane_clustering
            if 'hyperplane_clustering' in analysis_data:
                clustering_raw = analysis_data['hyperplane_clustering']
                
                # Iterate through all layers found
                for layer_name, layer_data in clustering_raw.items():
                    if not isinstance(layer_data, dict):
                        continue
                        
                    layer_info = {
                        'parameters': {},
                        'total_clusters': 0,
                        'total_noise': 0
                    }
                    
                    # Process each parameter type (weight, bias, etc.)
                    for param_name, param_data in layer_data.items():
                        if param_name in ['layer_name', 'clustering_params']:
                            continue  # Skip metadata fields
                            
                        if isinstance(param_data, dict) and 'param_data' in param_data:
                            clusters = []
                            
                            # Process each cluster
                            for cluster in param_data['param_data']:
                                cluster_info = {
                                    'cluster_label': cluster['cluster_label'],
                                    'size': cluster['size'],
                                    'run_ids': cluster['run_ids'],
                                    'centroid': [round(x, 3) for x in cluster['centroid']],
                                    'std': [round(x, 3) for x in cluster['std']]
                                }
                                clusters.append(cluster_info)
                            
                            layer_info['parameters'][param_name] = {
                                'clusters': clusters,
                                'n_clusters': param_data.get('n_clusters', 0),
                                'noise_count': param_data.get('noise_count', 0)
                            }
                            
                            # Accumulate totals for layer summary
                            layer_info['total_clusters'] += param_data.get('n_clusters', 0)
                            layer_info['total_noise'] += param_data.get('noise_count', 0)
                    
                    clustering_data['layers'][layer_name] = layer_info
                        
        except (KeyError, TypeError) as e:
            # Return empty clustering if data is missing or malformed
            pass
            
        return clustering_data
        
    def _load_mirror_weights_data(self, analysis_data: Optional[Dict]) -> Dict[str, Any]:
        """Load mirror weights analysis data."""
        mirror_data = {'summary': {}, 'runs': []}
        
        if not analysis_data:
            return mirror_data
            
        try:
            if 'mirror_weights' in analysis_data:
                mirror_weights = analysis_data['mirror_weights']
                
                if mirror_weights:  # Check if we have data
                    # Collect all cosine similarities for summary stats
                    all_cosines = []
                    total_pairs = 0
                    runs_with_mirrors = 0
                    
                    for run_data in mirror_weights:
                        run_id = run_data['run_id']
                        mirror_pairs = run_data.get('mirror_pairs', [])
                        mirror_count = run_data.get('mirror_count', 0)
                        
                        if mirror_count > 0:
                            runs_with_mirrors += 1
                            total_pairs += mirror_count
                            
                            # Parse cosine similarities from mirror pairs
                            for pair_str in mirror_pairs:
                                try:
                                    # Parse "(0, 1, -0.999)" format
                                    pair_tuple = ast.literal_eval(pair_str)
                                    if len(pair_tuple) >= 3:
                                        cosine = float(pair_tuple[2])
                                        all_cosines.append(cosine)
                                except (ValueError, SyntaxError):
                                    # Skip malformed pairs
                                    pass
                        
                        mirror_data['runs'].append({
                            'run_id': run_id,
                            'mirror_pairs': mirror_pairs,
                            'mirror_count': mirror_count
                        })
                    
                    # Compute summary statistics
                    if all_cosines:
                        mirror_data['summary'] = {
                            'total_runs_analyzed': len(mirror_weights),
                            'runs_with_mirrors': runs_with_mirrors,
                            'total_pairs': total_pairs,
                            'mean_cosine': round(np.mean(all_cosines), 4),
                            'std_cosine': round(np.std(all_cosines), 4),
                            'min_cosine': round(min(all_cosines), 4),
                            'max_cosine': round(max(all_cosines), 4)
                        }
                    else:
                        mirror_data['summary'] = {
                            'total_runs_analyzed': len(mirror_weights),
                            'runs_with_mirrors': 0,
                            'total_pairs': 0
                        }
                        
        except (KeyError, TypeError):
            # Return empty mirror data if missing or malformed
            pass
            
        return mirror_data
        
    def _update_overview_panel(self, data: Optional[Dict[str, Any]]) -> None:
        """Update the overview panel with experiment data."""
        if data is None:
            self.description_var.set("No experiment loaded")
            self.run_count_var.set("—")
            self.last_updated_var.set("—")
            self._update_accuracy_panel({})
            self._update_epochs_panel({})
            self._update_clustering_panel({})
            self._update_mirror_weights_panel({})
        else:
            self.description_var.set(data.get('description', 'N/A'))
            self.run_count_var.set(str(data.get('num_runs', 'N/A')))
            self.last_updated_var.set(data.get('last_updated', 'N/A'))
            self._update_accuracy_panel(data.get('accuracy', {}))
            self._update_epochs_panel(data.get('epochs', {}))
            self._update_clustering_panel(data.get('clustering', {}))
            self._update_mirror_weights_panel(data.get('mirror_weights', {}))
            
    def _update_accuracy_panel(self, accuracy_data: Dict[str, Any]) -> None:
        """Update the accuracy buckets display."""
        counts = accuracy_data.get('counts', {})
        run_ids = accuracy_data.get('run_ids', {})
        
        # Check for count mismatch and warn in status bar
        if counts and hasattr(self, 'run_count_var'):
            total_from_buckets = sum(counts.values())
            try:
                total_runs = int(self.run_count_var.get())
                if total_from_buckets != total_runs:
                    self.status_var.set(f"Warning: Accuracy bucket total ({total_from_buckets}) != total runs ({total_runs})")
            except (ValueError, TypeError):
                pass  # Skip validation if run count isn't a number
        
        for level in [0, 25, 50, 75, 100]:
            widgets = self.accuracy_widgets[level]
            
            # Update count
            count = counts.get(level, 0) if counts else 0
            widgets['count_var'].set(str(count))
            
            # Update run IDs availability and data
            level_run_ids = run_ids.get(level, []) if run_ids else []
            widgets['runs_data'] = level_run_ids
            
            # Enable/disable the show runs button
            has_run_ids = bool(level_run_ids) or (count > 0 and run_ids)  # Enable if we have run IDs or could have them
            if not run_ids:  # No run IDs available at all
                widgets['runs_button'].configure(state='disabled')
                if count > 0:
                    widgets['runs_button'].configure(text="(run IDs unavailable)")
                else:
                    widgets['runs_button'].configure(text="Show runs")
            else:
                widgets['runs_button'].configure(state='normal', text="Show runs")
                
            # Reset toggle state
            widgets['runs_var'].set(False)
            widgets['runs_label'].pack_forget()
            
    def _update_epochs_panel(self, epochs_data: Dict[str, Any]) -> None:
        """Update the epoch percentiles display."""
        percentiles = epochs_data.get('percentiles', {})
        
        for pct_label in ['0th', '10th', '25th', '50th', '75th', '90th', '100th']:
            if percentiles and pct_label in percentiles:
                value = str(percentiles[pct_label])
            else:
                value = "—"
            self.epochs_widgets[pct_label].set(value)
            
    def _toggle_runs_display(self, level: int) -> None:
        """Toggle the display of run IDs for an accuracy bucket."""
        widgets = self.accuracy_widgets[level]
        is_shown = widgets['runs_var'].get()
        
        if is_shown and widgets['runs_data']:
            # Show the run IDs
            run_ids_str = ", ".join(map(str, widgets['runs_data']))
            widgets['runs_label'].configure(text=f"Runs: {run_ids_str}")
            widgets['runs_label'].pack(side="left", padx=(8, 0))
        else:
            # Hide the run IDs
            widgets['runs_label'].pack_forget()
            
    def _update_clustering_panel(self, clustering_data: Dict[str, Any]) -> None:
        """Update the clustering analysis display."""
        # Clear existing content
        for widget in self.clustering_container.winfo_children():
            widget.destroy()
            
        layers = clustering_data.get('layers', {})
        
        if not layers:
            # Create new status label since we destroyed all widgets
            status_label = ttk.Label(self.clustering_container, text="No clustering analysis available", foreground="#666")
            status_label.pack()
            return
       
        # Create expandable sections for each layer (no internal scrolling needed)
        for layer_name, layer_info in layers.items():
            # Layer summary frame
            layer_frame = ttk.LabelFrame(self.clustering_container, text=f"{layer_name}", padding=(4, 4))
            layer_frame.pack(fill="x", pady=2, padx=2)
            
            # Summary info
            total_clusters = layer_info.get('total_clusters', 0)
            total_noise = layer_info.get('total_noise', 0)
            summary_text = f"{total_clusters} clusters"
            if total_noise > 0:
                summary_text += f", {total_noise} noise points"
                
            summary_label = ttk.Label(layer_frame, text=summary_text, foreground="#333")
            summary_label.pack(anchor="w")
            
            # Parameters details
            parameters = layer_info.get('parameters', {})
            for param_name, param_info in parameters.items():
                param_frame = ttk.Frame(layer_frame)
                param_frame.pack(fill="x", padx=(16, 0), pady=2)
                
                # Parameter header with summary
                n_clusters = param_info.get('n_clusters', 0)
                noise_count = param_info.get('noise_count', 0)
                header_text = f"{param_name}: {n_clusters} clusters"
                if noise_count > 0:
                    header_text += f", {noise_count} noise"
                    
                param_header = ttk.Label(param_frame, text=header_text, font=("TkDefaultFont", 9, "bold"))
                param_header.pack(anchor="w")
                
                # Clusters details
                clusters = param_info.get('clusters', [])
                if clusters:
                    for i, cluster in enumerate(clusters):
                        cluster_frame = ttk.Frame(param_frame)
                        cluster_frame.pack(fill="x", padx=(8, 0), pady=1)
                        
                        # Cluster basic info
                        cluster_label = cluster['cluster_label']
                        size = cluster['size']
                        run_count = len(cluster['run_ids'])
                        
                        basic_info = f"  Cluster {cluster_label}: {size} points, {run_count} runs"
                        basic_label = ttk.Label(cluster_frame, text=basic_info, font=("TkDefaultFont", 8))
                        basic_label.pack(anchor="w")
                        
                        # Centroid and std on separate lines for readability
                        centroid = cluster['centroid']
                        std = cluster['std']
                        
                        # Format arrays more compactly
                        if len(centroid) <= 5:
                            centroid_str = f"[{', '.join(f'{x:.3f}' for x in centroid)}]"
                            std_str = f"[{', '.join(f'{x:.3f}' for x in std)}]"
                        else:
                            # Truncate long arrays
                            centroid_str = f"[{', '.join(f'{x:.3f}' for x in centroid[:3])}...] ({len(centroid)}D)"
                            std_str = f"[{', '.join(f'{x:.3f}' for x in std[:3])}...] ({len(std)}D)"
                        
                        details_text = f"    Centroid: {centroid_str}  |  Std: {std_str}"
                        details_label = ttk.Label(cluster_frame, text=details_text, 
                                                font=("Consolas", 8), foreground="#666")
                        details_label.pack(anchor="w")
                        
                        # Run IDs - show first few and count if too many
                        run_ids = cluster['run_ids']
                        unique_runs = sorted(set(run_ids))  # Remove duplicates and sort
                        
                        if len(unique_runs) <= 10:
                            runs_str = f"    Runs: {', '.join(map(str, unique_runs))}"
                        else:
                            first_few = ', '.join(map(str, unique_runs[:8]))
                            runs_str = f"    Runs: {first_few}... ({len(unique_runs)} unique)"
                        
                        runs_label = ttk.Label(cluster_frame, text=runs_str, 
                                            font=("TkDefaultFont", 8), foreground="#888")
                        runs_label.pack(anchor="w")
        
    def _update_mirror_weights_panel(self, mirror_data: Dict[str, Any]) -> None:
        """Update the mirror weights analysis display."""
        print(mirror_data)
        summary = mirror_data.get('summary', {})
        runs = mirror_data.get('runs', [])
        
        if not summary:
            self.mirror_summary_var.set("No mirror weights data available")
            self.mirror_details_button.configure(state='disabled')
            self.mirror_details_var.set(False)
            self.mirror_details_frame.pack_forget()
            return
            
        # Update summary display
        total_runs = summary.get('total_runs_analyzed', 0)
        runs_with_mirrors = summary.get('runs_with_mirrors', 0)
        total_pairs = summary.get('total_pairs', 0)
        
        summary_text = f"Analyzed {total_runs} runs: {runs_with_mirrors} with mirrors ({total_pairs} pairs)"
        
        if 'mean_cosine' in summary:
            mean_cos = summary['mean_cosine']
            std_cos = summary['std_cosine']
            summary_text += f", mean cosine: {mean_cos:.4f} ± {std_cos:.4f}"
            
        self.mirror_summary_var.set(summary_text)
        self.mirror_details_button.configure(state='normal')
        
        # Prepare details text
        if runs:
            details_lines = []
            for run_data in runs:
                run_id = run_data['run_id']
                mirror_count = run_data['mirror_count']
                
                if mirror_count > 0:
                    pairs_text = ", ".join(run_data['mirror_pairs'])
                    details_lines.append(f"Run {run_id}: {pairs_text}")
                    
            self.mirror_details_content = "\n".join(details_lines) if details_lines else "No mirror pairs found"
        else:
            self.mirror_details_content = "No run data available"
            
    def _toggle_mirror_details(self) -> None:
        """Toggle the display of mirror weights details."""
        if self.mirror_details_var.get():
            # Show details
            self.mirror_details_frame.pack(fill="both", expand=True, pady=(4, 0))
            self.mirror_details_text.delete(1.0, tk.END)
            self.mirror_details_text.insert(1.0, getattr(self, 'mirror_details_content', 'No details available'))
            self.mirror_details_text.configure(state='disabled')
        else:
            # Hide details
            self.mirror_details_frame.pack_forget()

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

    def _on_mousewheel(self, event):
        """Handle mousewheel scrolling."""
        try:
            # Find the main canvas
            for child in self.winfo_children():
                if isinstance(child, tk.Canvas):
                    child.yview_scroll(int(-1*(event.delta/120)), "units")
                    break
        except:
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