"""
Centralized logging and terminal UI utilities for the EEG pipeline.

This module handles:
- Rich terminal UI setup (progress bars, live logs)
- Python logging configuration
- Capturing and routing MNE library output
- Custom log messaging with timestamp and elapsed time tracking
"""

import logging
import io
import contextlib
import time
from collections import deque

# Rich terminal UI components (optional; graceful fallback if not available)
try:
    from rich.console import Console
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    Live = None
    Progress = None
    BarColumn = None
    TextColumn = None
    TimeElapsedColumn = None
    TimeRemainingColumn = None
    Panel = None
    console = None

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Circular buffer of the last 20 log lines (for live display)
LOG_LINES = deque(maxlen=20)

# Script start time for elapsed time calculations
START_TIME = time.time()


# ============================================================================
# LOGGING HANDLERS AND FORMATTERS
# ============================================================================

class RichLogHandler(logging.Handler):
    """
    Custom logging handler that appends formatted messages to LOG_LINES.
    
    This handler feeds messages into the Rich live log panel and avoids
    consecutive duplicate lines.
    """
    
    def emit(self, record):
        """Format and append a log record to the buffer."""
        msg = self.format(record).strip()
        # Avoid consecutive duplicate lines
        if len(LOG_LINES) == 0 or LOG_LINES[-1] != msg:
            LOG_LINES.append(msg)


def init_logging():
    """
    Initialize Python logging with the custom RichLogHandler.
    
    This sets up the root logger to route all logs through our custom handler,
    and disables verbose MNE logging to prevent duplicate/noisy output.
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Attach our custom handler
    rich_handler = RichLogHandler()
    rich_handler.setLevel(logging.INFO)
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(rich_handler)
    
    # Disable MNE logging to prevent verbose/duplicate output
    try:
        import mne
        try:
            mne.set_log_level("ERROR")
        except Exception:
            pass
        mne_logger = logging.getLogger("mne")
        try:
            mne_logger.handlers = []
            mne_logger.propagate = False
            mne_logger.setLevel(logging.ERROR)
            mne_logger.disabled = True
        except Exception:
            pass
    except ImportError:
        pass


def terminal_log(msg: str, level: str = "info"):
    """
    Log a message to the terminal and Rich log panel.
    
    This is the main interface for emitting log messages throughout the pipeline.
    Messages are routed through Python's logging system and captured by RichLogHandler.
    
    Args:
        msg (str): The message to log
        level (str): Log level ('debug', 'info', 'warning', 'error')
    """
    if level == "debug":
        logging.debug(msg)
    elif level == "warning":
        logging.warning(msg)
    elif level == "error":
        logging.error(msg)
    else:
        logging.info(msg)


# ============================================================================
# CONTEXT MANAGERS FOR OUTPUT CAPTURE
# ============================================================================

@contextlib.contextmanager
def redirect_streams():
    """
    Context manager to capture stdout/stderr and discard MNE verbose output.
    
    This silently suppresses library-level verbose output (e.g., from MNE's
    BDF reading, filtering, or event finding) so that only custom terminal_log
    messages appear in the log panel.
    
    Usage:
        with redirect_streams():
            raw = mne.io.read_raw_bdf(...)  # MNE output is suppressed
        terminal_log("BDF file loaded")  # Custom message appears instead
    """
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    
    # Temporarily disable MNE logger
    try:
        import mne
        mne_logger = logging.getLogger("mne")
        prev_level = mne_logger.level
        mne_logger.setLevel(logging.ERROR)
        if hasattr(mne, "set_log_level"):
            try:
                mne.set_log_level("ERROR")
            except Exception:
                pass
    except ImportError:
        mne = None
        prev_level = None
    
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            yield
    finally:
        # Restore MNE logging settings
        try:
            if mne is not None:
                mne_logger.setLevel(prev_level if prev_level is not None else logging.INFO)
                if hasattr(mne, "set_log_level") and prev_level is not None:
                    try:
                        mne.set_log_level("INFO")
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Discard captured output (we provide custom messages instead)
    try:
        _ = buf_out.getvalue()
        _ = buf_err.getvalue()
    except Exception:
        pass


# ============================================================================
# RICH TERMINAL UI COMPONENTS
# ============================================================================

class LogRenderable:
    """
    Rich renderable that displays the circular log buffer as a Panel.
    
    This is used with Rich's Live display to show a real-time log panel
    that updates as new messages are logged.
    """
    
    def __rich_console__(self, console, options):
        """Render the log panel with current buffer contents."""
        content = "\n".join(list(LOG_LINES))
        yield Panel(content, title="Logs", width=80)


def get_progress_bar():
    """
    Create a Rich Progress bar for tracking long-running tasks.
    
    Returns:
        Progress: A configured progress bar, or None if Rich is not available
    """
    if not RICH_AVAILABLE or Progress is None:
        return None
    
    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        "{task.completed}/{task.total}",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    )


def get_live_display():
    """
    Create a Rich Live display for the log panel.
    
    Returns:
        Live: A live display object, or None if Rich is not available
    """
    if not RICH_AVAILABLE or Live is None:
        return None
    
    return Live(LogRenderable(), console=console, refresh_per_second=5)
