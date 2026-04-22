"""
sidebar_nav.py — Sidebar Navigation Widget for PixelPaws
=========================================================
Drop-in replacement for ttk.Notebook that uses a vertical sidebar
with workflow-grouped navigation items.

Provides a Notebook-compatible API (add, select, tab, bind) so
existing tab creation code works unchanged.
"""

import tkinter as tk

try:
    import ttkbootstrap as ttk
    _TTKBOOTSTRAP = True
except ImportError:
    from tkinter import ttk
    _TTKBOOTSTRAP = False


# ---------------------------------------------------------------------------
# Theme colour palettes
# ---------------------------------------------------------------------------

_COLORS = {
    'light': {
        'sidebar_bg': '#ffffff',
        'active_bg': '#e8f0fe',
        'hover_bg': '#f0f0f0',
        'accent': '#0d6efd',
        'group_fg': '#888888',
        'item_fg': '#222222',
        'title_fg': '#333333',
        'sep': '#e0e0e0',
    },
    'dark': {
        'sidebar_bg': '#2b2b2b',
        'active_bg': '#3a4a5a',
        'hover_bg': '#353535',
        'accent': '#5dabf7',
        'group_fg': '#999999',
        'item_fg': '#e0e0e0',
        'title_fg': '#e0e0e0',
        'sep': '#444444',
    },
}


def _get_theme_colors(mode='light'):
    """Return the colour dict for *mode*.

    Uses curated hardcoded values for sidebar_bg and active_bg (the
    ttkbootstrap palette colours look jarring), but still pulls primary
    and fg from the live theme so accent/text stay consistent.
    """
    base = _COLORS.get(mode, _COLORS['light']).copy()
    if _TTKBOOTSTRAP:
        try:
            style = ttk.Style()
            colors = style.colors
            base['accent'] = str(colors.primary)
            base['item_fg'] = str(colors.fg)
            base['title_fg'] = str(colors.fg)
        except Exception:
            pass
    return base


# ---------------------------------------------------------------------------
# SidebarNav
# ---------------------------------------------------------------------------

class SidebarNav(ttk.Frame):
    """A sidebar-based navigation widget with Notebook-compatible API.

    Parameters
    ----------
    parent : tk widget
        The parent widget (typically ``root``).
    groups : dict[str, list[str]]
        Ordered mapping from group name to list of tab text labels.
        Example::

            {"Train & Label": ["🎓 Train Classifier", "🧠 Active Learning"],
             "Predict & Evaluate": ["🎬 Predict", "📊 Evaluate", "📦 Batch"],
             ...}
    width : int
        Sidebar width in pixels (default 220).
    collapsed_width : int
        Width when collapsed to icon-only mode (default 44).
    """

    def __init__(self, parent, groups=None, width=220, collapsed_width=44):
        super().__init__(parent)
        self._groups = groups or {}
        self._sidebar_width = width
        self._collapsed_width = collapsed_width
        self._collapsed = False
        self._mode = 'light'

        # Storage
        self._pages = {}          # tab_text -> {'frame': widget, 'nav': Frame, 'group': str}
        self._page_order = []     # tab_text list in add() order
        self._active_key = None
        self._callbacks = {}      # event sequence -> [callbacks]
        self._nav_widgets = {}    # tab_text -> nav item Frame
        self._group_labels = {}   # group_name -> Label widget
        self._status_dots = {}    # tab_text -> Canvas dot widget
        self._text_labels = {}    # tab_text -> Label (for collapse toggling)

        self._colors = _get_theme_colors(self._mode)

        # --- Layout: sidebar | separator | content ---
        self._sidebar = tk.Frame(self, width=width, bg=self._colors['sidebar_bg'])
        self._sidebar.pack(side='left', fill='y')
        self._sidebar.pack_propagate(False)

        self._sep = ttk.Separator(self, orient='vertical')
        self._sep.pack(side='left', fill='y')

        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(side='left', fill='both', expand=True)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        self._hover_active = False  # suppress hover during construction

        # --- Build sidebar contents ---
        self._build_sidebar()

        # Enable hover after construction to avoid spurious Enter events
        self.after(200, self._enable_hover)

        # Reset stuck hovers when mouse truly leaves the sidebar
        self._sidebar.bind('<Leave>', self._on_sidebar_leave)

    # ------------------------------------------------------------------
    # Sidebar construction
    # ------------------------------------------------------------------

    def _build_sidebar(self):
        """Create sidebar header, groups, and placeholder nav items."""
        c = self._colors

        # App title
        title = tk.Label(self._sidebar, text="PixelPaws",
                         bg=c['sidebar_bg'], fg=c['title_fg'],
                         font=('Arial', 13, 'bold'), anchor='w')
        title.pack(fill='x', padx=12, pady=(14, 4))
        self._title_label = title

        # Thin separator
        sep = tk.Frame(self._sidebar, height=1, bg=c['sep'])
        sep.pack(fill='x', padx=8, pady=(4, 10))
        self._title_sep = sep

        # Build groups with placeholder nav items
        self._nav_container = tk.Frame(self._sidebar, bg=c['sidebar_bg'])
        self._nav_container.pack(fill='both', expand=True)

        for group_name, item_labels in self._groups.items():
            self._build_group(group_name, item_labels)

    def _build_group(self, group_name, item_labels):
        """Create a group header and nav items for each label."""
        c = self._colors
        container = self._nav_container

        # Group header
        grp = tk.Label(container, text=group_name.upper(),
                       bg=c['sidebar_bg'], fg=c['group_fg'],
                       font=('Arial', 8, 'bold'), anchor='w')
        grp.pack(fill='x', padx=14, pady=(12, 2))
        self._group_labels[group_name] = grp

        # Nav items (placeholders — frames will be registered on add())
        for text in item_labels:
            self._build_nav_item(container, text)

    def _build_nav_item(self, container, text):
        """Create a single nav item row."""
        c = self._colors

        # Outer frame for the row
        row = tk.Frame(container, bg=c['sidebar_bg'], cursor='hand2')
        row.pack(fill='x', padx=4, pady=1)

        # Accent bar (left border, 3px, hidden by default)
        accent = tk.Frame(row, width=3, bg=c['sidebar_bg'])
        accent.pack(side='left', fill='y')

        # Emoji + text
        # Split text into emoji prefix and label
        parts = text.split(' ', 1)
        emoji = parts[0] if len(parts) > 1 else ''
        label_text = parts[1] if len(parts) > 1 else text

        emoji_lbl = tk.Label(row, text=emoji, bg=c['sidebar_bg'],
                             font=('Arial', 11), width=2, anchor='center')
        emoji_lbl.pack(side='left', padx=(4, 2), pady=6)

        text_lbl = tk.Label(row, text=label_text, bg=c['sidebar_bg'],
                            fg=c['item_fg'], font=('Arial', 10), anchor='w')
        text_lbl.pack(side='left', fill='x', expand=True, pady=6)
        self._text_labels[text] = text_lbl

        # Status dot (small canvas, hidden by default)
        dot = tk.Canvas(row, width=8, height=8, bg=c['sidebar_bg'],
                        highlightthickness=0)
        dot.pack(side='right', padx=(0, 8))
        self._status_dots[text] = dot

        # Store references
        self._nav_widgets[text] = {
            'row': row, 'accent': accent,
            'emoji': emoji_lbl, 'text_lbl': text_lbl, 'dot': dot,
        }

        # Bind click — on all sub-widgets so clicks anywhere in the row work
        for widget in (row, accent, emoji_lbl, text_lbl):
            widget.bind('<Button-1>', lambda e, t=text: self._on_nav_click(t))

        # Hover — Enter clears all then highlights; Leave checks containment
        for widget in (row, accent, emoji_lbl, text_lbl):
            widget.bind('<Enter>', lambda e, t=text: self._on_hover_enter(t))
            widget.bind('<Leave>', lambda e, t=text: self._on_hover_leave(t))

    # ------------------------------------------------------------------
    # Notebook-compatible API
    # ------------------------------------------------------------------

    def add(self, frame, text="", **kw):
        """Register a content frame under the given tab text.

        The frame is placed in the content area via grid stacking.
        The first frame added becomes the active (visible) page.
        """
        self._pages[text] = {'frame': frame, 'group': self._find_group(text)}
        self._page_order.append(text)

        # Grid the frame in content area (stacked)
        frame.grid(in_=self.content_frame, row=0, column=0, sticky='nsew')

        # Show the first page added
        if self._active_key is None:
            self._active_key = text
            frame.tkraise()
            self._highlight_nav(text)
        else:
            # Lower it behind the active frame
            pass

    def select(self, tab_id=None):
        """Get or set the active page.

        If *tab_id* is None, return the text key of the active page.
        Otherwise, switch to the page identified by *tab_id* (text string
        or frame widget).
        """
        if tab_id is None:
            return self._active_key

        # Resolve tab_id to a text key
        key = self._resolve_key(tab_id)
        if key and key in self._pages:
            self._switch_to(key)
        return key

    def tab(self, tab_id, option=None):
        """Mimic ``notebook.tab(tab_id, option)``.

        Returns the text label or a metadata dict.
        """
        key = self._resolve_key(tab_id)
        info = {'text': key}
        if option == 'text':
            return key
        return info

    def bind(self, sequence, callback, add=None):
        """Bind a callback. Supports ``<<NotebookTabChanged>>``."""
        if sequence not in self._callbacks:
            self._callbacks[sequence] = []
        self._callbacks[sequence].append(callback)

    # ------------------------------------------------------------------
    # Sidebar-specific API
    # ------------------------------------------------------------------

    def hide_item(self, tab_text):
        """Hide a nav item (e.g. when its module is unavailable)."""
        widgets = self._nav_widgets.get(tab_text)
        if widgets:
            widgets['row'].pack_forget()

    def set_status(self, tab_text, status):
        """Set a status indicator on a nav item.

        *status*: 'complete' (green), 'in_progress' (yellow),
                  'error' (red), or None (hidden).
        """
        dot = self._status_dots.get(tab_text)
        if not dot:
            return
        dot.delete('all')
        color_map = {
            'complete': '#28a745',
            'in_progress': '#ffc107',
            'error': '#dc3545',
        }
        if status and status in color_map:
            dot.create_oval(1, 1, 7, 7, fill=color_map[status], outline='')
        dot.configure(bg=self._nav_widgets[tab_text]['row'].cget('bg'))

    def toggle_collapse(self):
        """Toggle between full sidebar and icon-only mode."""
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._sidebar.config(width=self._collapsed_width)
            self._title_label.pack_forget()
            self._title_sep.pack_forget()
            # Hide text labels, group headers
            for lbl in self._group_labels.values():
                lbl.pack_forget()
            for text, widgets in self._nav_widgets.items():
                widgets['text_lbl'].pack_forget()
                widgets['dot'].pack_forget()
        else:
            self._sidebar.config(width=self._sidebar_width)
            # Rebuild sidebar contents
            for w in self._nav_container.winfo_children():
                w.destroy()
            self._nav_widgets.clear()
            self._group_labels.clear()
            self._status_dots.clear()
            self._text_labels.clear()
            self._title_label.pack(fill='x', padx=12, pady=(14, 4),
                                   before=self._title_sep)
            self._title_sep.pack(fill='x', padx=8, pady=(4, 10),
                                 before=self._nav_container)
            for group_name, item_labels in self._groups.items():
                self._build_group(group_name, item_labels)
            if self._active_key:
                self._highlight_nav(self._active_key)

    def update_theme(self, mode):
        """Update sidebar colors for 'light' or 'dark' mode."""
        self._mode = mode
        self._colors = _get_theme_colors(mode)
        c = self._colors

        self._sidebar.config(bg=c['sidebar_bg'])
        self._nav_container.config(bg=c['sidebar_bg'])
        self._title_label.config(bg=c['sidebar_bg'], fg=c['title_fg'])
        self._title_sep.config(bg=c['sep'])

        for lbl in self._group_labels.values():
            lbl.config(bg=c['sidebar_bg'], fg=c['group_fg'])

        for text, widgets in self._nav_widgets.items():
            is_active = (text == self._active_key)
            bg = c['active_bg'] if is_active else c['sidebar_bg']
            accent_bg = c['accent'] if is_active else c['sidebar_bg']
            for w in (widgets['row'], widgets['emoji'],
                      widgets['text_lbl'], widgets['dot']):
                w.config(bg=bg)
            widgets['accent'].config(bg=accent_bg)
            widgets['text_lbl'].config(fg=c['item_fg'])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _find_group(self, text):
        """Find which group a tab text belongs to."""
        for group_name, items in self._groups.items():
            if text in items:
                return group_name
        return None

    def _resolve_key(self, tab_id):
        """Resolve a tab_id (text string or frame widget) to a text key."""
        if isinstance(tab_id, str):
            if tab_id in self._pages:
                return tab_id
            # Could be a tkinter widget path — search by frame
            for key, info in self._pages.items():
                if str(info['frame']) == tab_id:
                    return key
            return tab_id
        # Might be a frame widget
        for key, info in self._pages.items():
            if info['frame'] is tab_id:
                return key
        return None

    def _switch_to(self, key):
        """Switch to the page identified by *key*."""
        if key == self._active_key:
            return
        old_key = self._active_key
        self._active_key = key

        # Raise the new frame
        self._pages[key]['frame'].tkraise()

        # Update sidebar highlights
        if old_key:
            self._unhighlight_nav(old_key)
        self._highlight_nav(key)

        # Fire callbacks
        for cb in self._callbacks.get('<<NotebookTabChanged>>', []):
            try:
                cb(None)
            except Exception:
                pass

    def _on_nav_click(self, text):
        """Handle click on a nav item."""
        if text in self._pages:
            self._switch_to(text)

    def _highlight_nav(self, text):
        """Visually highlight the active nav item."""
        widgets = self._nav_widgets.get(text)
        if not widgets:
            return
        c = self._colors
        for w in (widgets['row'], widgets['emoji'],
                  widgets['text_lbl'], widgets['dot']):
            w.config(bg=c['active_bg'])
        widgets['accent'].config(bg=c['accent'])
        widgets['text_lbl'].config(font=('Arial', 10, 'bold'))

    def _unhighlight_nav(self, text):
        """Remove highlight from a nav item."""
        widgets = self._nav_widgets.get(text)
        if not widgets:
            return
        c = self._colors
        for w in (widgets['row'], widgets['emoji'],
                  widgets['text_lbl'], widgets['dot']):
            w.config(bg=c['sidebar_bg'])
        widgets['accent'].config(bg=c['sidebar_bg'])
        widgets['text_lbl'].config(font=('Arial', 10))

    def _enable_hover(self):
        """Called after construction delay to enable hover effects."""
        self._hover_active = True

    def _on_hover_enter(self, text):
        """Handle mouse entering a nav item.

        Clears ALL previous hover highlights first, then applies hover
        to the new item.  This avoids stale highlights caused by missed
        ``<Leave>`` events between nested child widgets.
        """
        if not self._hover_active:
            return
        # Reset every non-active item first
        self._clear_all_hovers()
        if text == self._active_key:
            return
        widgets = self._nav_widgets.get(text)
        if not widgets:
            return
        c = self._colors
        for w in (widgets['row'], widgets['emoji'],
                  widgets['text_lbl'], widgets['dot']):
            w.config(bg=c['hover_bg'])
        widgets['accent'].config(bg=c['hover_bg'])

    def _on_hover_leave(self, text):
        """Handle mouse leaving a nav item — check containment."""
        if not self._hover_active:
            return
        if text == self._active_key:
            return
        widgets = self._nav_widgets.get(text)
        if not widgets:
            return
        row = widgets['row']
        try:
            mx, my = row.winfo_pointerxy()
            rx, ry = row.winfo_rootx(), row.winfo_rooty()
            if rx <= mx < rx + row.winfo_width() and ry <= my < ry + row.winfo_height():
                return  # Mouse still inside this row (crossed to child widget)
        except Exception:
            pass
        c = self._colors
        for w in (row, widgets['emoji'], widgets['text_lbl'], widgets['dot']):
            w.config(bg=c['sidebar_bg'])
        widgets['accent'].config(bg=c['sidebar_bg'])

    def _on_sidebar_leave(self, event):
        """Clear hovers only when the mouse truly leaves the sidebar."""
        if not self._hover_active:
            return
        try:
            w = self._sidebar.winfo_containing(event.x_root, event.y_root)
            if w and (w is self._sidebar
                      or str(w).startswith(str(self._sidebar))):
                return  # Still inside sidebar, just crossing child boundaries
        except Exception:
            pass
        self._clear_all_hovers()

    def _clear_all_hovers(self):
        """Reset all non-active items to default bg."""
        c = self._colors
        for text, widgets in self._nav_widgets.items():
            if text == self._active_key:
                continue
            for w in (widgets['row'], widgets['emoji'],
                      widgets['text_lbl'], widgets['dot']):
                w.config(bg=c['sidebar_bg'])
            widgets['accent'].config(bg=c['sidebar_bg'])
