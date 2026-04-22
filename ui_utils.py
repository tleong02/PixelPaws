"""
ui_utils.py — Shared UI utilities for PixelPaws
=================================================
Theme-aware ToolTip, canvas helpers, and mousewheel binding.
"""

import tkinter as tk


# ============================================================================
# Theme-aware ToolTip
# ============================================================================

class ToolTip:
    """Tooltip that adapts to light/dark themes."""

    # Class-level theme colours; call ToolTip.set_theme() to update.
    _bg = '#ffffcc'
    _fg = '#000000'

    @classmethod
    def set_theme(cls, mode: str):
        """Update tooltip colours for 'light' or 'dark' mode."""
        if mode == 'dark':
            cls._bg = '#3a3a3a'
            cls._fg = '#e0e0e0'
        else:
            cls._bg = '#ffffcc'
            cls._fg = '#000000'

    def __init__(self, widget, text):
        self._tip = None
        widget.bind('<Enter>', lambda e: self._show(widget, text))
        widget.bind('<Leave>', lambda e: self._hide())

    def _show(self, widget, text):
        x = widget.winfo_rootx() + 20
        y = widget.winfo_rooty() + widget.winfo_height() + 4
        self._tip = tk.Toplevel(widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f'+{x}+{y}')
        tk.Label(self._tip, text=text, background=self._bg,
                 foreground=self._fg, relief='solid', borderwidth=1,
                 font=('Arial', 9), wraplength=320,
                 justify='left').pack(ipadx=4, ipady=2)

    def _hide(self):
        if self._tip:
            self._tip.destroy()
            self._tip = None


# ============================================================================
# Canvas helpers (shared by analysis_tab, active_learning, unsupervised, etc.)
# ============================================================================

def _bind_tight_layout_on_resize(canvas, fig, rect=None):
    """Debounced redraw on widget resize so constrained_layout recomputes."""
    _timer = [None]
    widget = canvas.get_tk_widget()
    def _sync_size_and_draw():
        try:
            w, h = widget.winfo_width(), widget.winfo_height()
            if w > 1 and h > 1:
                fig.set_size_inches(w / fig.dpi, h / fig.dpi, forward=False)
            canvas.draw_idle()
        except Exception:
            pass
    def _on_configure(event):
        if _timer[0] is not None:
            widget.after_cancel(_timer[0])
        _timer[0] = widget.after(150, _sync_size_and_draw)
    widget.bind('<Configure>', _on_configure, add='+')
    widget.after(300, _sync_size_and_draw)


def _draw_canvas_fit(canvas, fig):
    """Force geometry computation, resize figure to widget, and draw."""
    widget = canvas.get_tk_widget()
    widget.update_idletasks()
    w, h = widget.winfo_width(), widget.winfo_height()
    if w > 1 and h > 1:
        fig.set_size_inches(w / fig.dpi, h / fig.dpi, forward=False)
    canvas.draw()


# ============================================================================
# Mousewheel scrolling for scrollable canvases
# ============================================================================

def bind_mousewheel(canvas):
    """Bind mousewheel scrolling to a tk.Canvas that uses yview for scroll.

    Binds on <Enter>/<Leave> so it only scrolls when the mouse is over
    the canvas area.  Works on Windows (MouseWheel) and Linux (Button-4/5).
    """
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    def _on_enter(event):
        canvas.bind_all('<MouseWheel>', _on_mousewheel)
        # Linux scroll events
        canvas.bind_all('<Button-4>', lambda e: canvas.yview_scroll(-3, 'units'))
        canvas.bind_all('<Button-5>', lambda e: canvas.yview_scroll(3, 'units'))

    def _on_leave(event):
        canvas.unbind_all('<MouseWheel>')
        canvas.unbind_all('<Button-4>')
        canvas.unbind_all('<Button-5>')

    canvas.bind('<Enter>', _on_enter)
    canvas.bind('<Leave>', _on_leave)
