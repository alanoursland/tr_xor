#!/usr/bin/env python3
"""
explorer.py — Results Explorer UI (stub)

This is the first step: open an application window.
We'll extend it to browse experiment folders (results/<experiment_name>)
and runs (runs/<run_id>) as we go.
"""

from __future__ import annotations

import os
import sys
import platform
import tkinter as tk
from tkinter import ttk


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
        self.minsize(600, 400)

        # Remember root path for later browsing logic
        self.results_root = RESULTS_ROOT

        self._build_menu()
        self._build_layout()
        self._apply_style()

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
        root = ttk.Frame(self, padding=(12, 12, 12, 8))
        root.pack(fill="both", expand=True)

        # Placeholder content (we'll replace with actual controls later)
        title = ttk.Label(root, text="Results Explorer (stub)", font=("Segoe UI", 16))
        subtitle = ttk.Label(
            root,
            text=f"results root: {self.results_root}",
            foreground="#666"
        )

        title.pack(anchor="w")
        subtitle.pack(anchor="w")

        # Stretchable spacer
        spacer = ttk.Frame(root)
        spacer.pack(fill="both", expand=True)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(8, 2))
        status.pack(side="bottom", fill="x")

    def _apply_style(self) -> None:
        style = ttk.Style(self)
        # Use a platform-appropriate theme
        try:
            if platform.system() == "Windows":
                style.theme_use("vista")
            elif platform.system() == "Darwin":
                style.theme_use("aqua")
            else:
                style.theme_use("clam")
        except tk.TclError:
            pass  # Fallback to default theme

    # --- Actions ---

    def _on_exit(self) -> None:
        self.destroy()

    def _show_about(self) -> None:
        from tkinter import messagebox
        messagebox.showinfo(
            "About",
            f"{APP_TITLE}\n\n"
            "A simple UI to browse experiment results and runs.\n"
            "We'll build the features step-by-step."
        )


def main(argv: list[str] | None = None) -> int:
    _configure_windows_dpi_awareness()
    app = ResultsExplorerApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
