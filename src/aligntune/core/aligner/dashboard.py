"""
AlignerDashboard: Rich terminal UI for interactive training.

Provides live monitoring dashboard with real-time metrics, examples,
and keyboard controls for training interaction.
"""

import logging
import threading
import time
from typing import Optional, Dict, Any

try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn, DownloadColumn
    from rich.console import Console
    from rich.sparklines import Sparkline
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Live = None
    Layout = None
    Panel = None
    Table = None
    Text = None
    Progress = None
    BarColumn = None
    TextColumn = None
    DownloadColumn = None
    Console = None
    Sparkline = None

logger = logging.getLogger(__name__)


class AlignerDashboard:
    """
    Terminal dashboard for interactive training inspection.

    Shows:
    - Reward/loss/KL progress
    - Recent training examples
    - Training status (step, elapsed, ETA)
    - Keyboard shortcuts help
    """

    def __init__(self, aligner_session, refresh_interval: float = 0.5):
        """
        Initialize dashboard.

        Args:
            aligner_session: AlignerSession instance to monitor
            refresh_interval: Update frequency in seconds
        """
        if not RICH_AVAILABLE:
            raise ImportError("rich library required for dashboard. Install with: pip install rich")

        self.aligner_session = aligner_session
        self.refresh_interval = refresh_interval
        self.console = Console()
        self._running = False
        self._dashboard_thread: Optional[threading.Thread] = None
        self._paused_notification_shown = False

        logger.info("AlignerDashboard initialized")

    def start(self) -> None:
        """Start dashboard update loop."""
        if self._dashboard_thread is not None and self._dashboard_thread.is_alive():
            logger.warning("Dashboard already running")
            return

        self._running = True
        self._dashboard_thread = threading.Thread(
            target=self._update_loop,
            daemon=True,
            name="AlignerDashboardThread"
        )
        self._dashboard_thread.start()
        logger.info("Dashboard started")

    def stop(self) -> None:
        """Stop dashboard update loop."""
        self._running = False
        if self._dashboard_thread:
            self._dashboard_thread.join(timeout=5)
        logger.info("Dashboard stopped")

    def _update_loop(self) -> None:
        """Main dashboard update loop."""
        try:
            layout = Layout()
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="main"),
                Layout(name="footer", size=2),
            )

            with Live(layout, refresh_per_second=2, console=self.console) as live:
                while self._running:
                    try:
                        # Update layout
                        layout["header"].update(self._make_header())
                        layout["main"].split_row(
                            Layout(self._make_metrics_panel()),
                            Layout(self._make_examples_panel()),
                        )
                        layout["footer"].update(self._make_footer())

                        time.sleep(self.refresh_interval)
                    except Exception as e:
                        logger.debug(f"Dashboard update error: {e}")

        except Exception as e:
            logger.error(f"Dashboard loop failed: {e}")

    def _make_header(self) -> Panel:
        """Create header panel with title and status."""
        state = self.aligner_session.peek()
        status_text = ""

        if self.aligner_session.is_paused():
            status_text = "[bold red]PAUSED[/bold red]"
            self._paused_notification_shown = True
        elif self.aligner_session.is_training():
            status_text = "[bold green]TRAINING[/bold green]"
            self._paused_notification_shown = False
        else:
            status_text = "[bold yellow]IDLE[/bold yellow]"

        title = f"[bold cyan]AlignTune Aligner[/bold cyan] {status_text}"
        subtitle = f"Step: {state['step']} | Loss: {state['loss']:.4f} | Reward: {state['reward']:.4f}"

        return Panel(
            f"{title}\n{subtitle}",
            style="blue",
        )

    def _make_metrics_panel(self) -> Panel:
        """Create metrics panel with progress visualization."""
        state = self.aligner_session.peek()
        history = self.aligner_session.history()

        table = Table(show_header=False, box=None)
        table.add_column(style="cyan", width=20)
        table.add_column(style="magenta")

        # Loss
        table.add_row(
            "[bold]Loss[/bold]",
            f"{state['loss']:.6f}"
        )

        # Reward
        table.add_row(
            "[bold]Reward[/bold]",
            f"{state['reward']:.6f}"
        )

        # KL Divergence
        table.add_row(
            "[bold]KL Divergence[/bold]",
            f"{state['kl_divergence']:.6f}"
        )

        # Learning Rate
        table.add_row(
            "[bold]Learning Rate[/bold]",
            f"{state['learning_rate']:.2e}"
        )

        # Batch Size
        table.add_row(
            "[bold]Batch Size[/bold]",
            f"{state['batch_size']}"
        )

        # Elapsed Time
        table.add_row(
            "[bold]Elapsed Time[/bold]",
            f"{state['elapsed_time']:.1f}s"
        )

        # Sparkline for loss if history available
        if history.get("loss", []):
            loss_values = history["loss"][-20:]  # Last 20 points
            sparkline_text = "Loss trend: " + str(loss_values[-5:])  # Fallback
            try:
                sparkline = Sparkline(loss_values, height=1)
                table.add_row("[bold]Trend[/bold]", sparkline)
            except Exception:
                table.add_row("[bold]Trend[/bold]", sparkline_text)

        return Panel(table, title="[bold]Metrics[/bold]", expand=False)

    def _make_examples_panel(self) -> Panel:
        """Create panel with recent training examples."""
        worst = self.aligner_session.worst_examples(n=3)

        if not worst:
            return Panel(
                Text("No examples yet", style="dim"),
                title="[bold]Recent Examples[/bold]",
            )

        table = Table(show_header=True, box=None)
        table.add_column("Prompt", style="cyan", width=20)
        table.add_column("Chosen", style="green", width=20)
        table.add_column("Rejected", style="red", width=20)
        table.add_column("Δ Reward", style="magenta", width=10)

        for prompt, chosen, rejected, delta in worst:
            # Truncate long strings
            prompt_str = (prompt[:17] + "...") if len(prompt) > 20 else prompt
            chosen_str = (chosen[:17] + "...") if len(chosen) > 20 else chosen
            rejected_str = (rejected[:17] + "...") if len(rejected) > 20 else rejected

            delta_color = "green" if delta > 0 else "red"
            delta_str = f"[{delta_color}]{delta:+.4f}[/{delta_color}]"

            table.add_row(prompt_str, chosen_str, rejected_str, delta_str)

        return Panel(table, title="[bold]Worst Examples[/bold]")

    def _make_footer(self) -> Panel:
        """Create footer with keyboard shortcuts."""
        shortcuts = (
            "[bold cyan]Shortcuts:[/bold cyan] "
            "[yellow]P[/yellow]=Pause | "
            "[yellow]R[/yellow]=Resume | "
            "[yellow]S[/yellow]=Sample | "
            "[yellow]W[/yellow]=Worst | "
            "[yellow]Q[/yellow]=Quit"
        )

        return Panel(shortcuts, style="dim")

    def interactive_mode(self) -> None:
        """
        Run dashboard with interactive key handling.

        This is a simplified version - full keyboard handling would require
        additional libraries like `pynput` or custom signal handling.
        """
        self.console.print(
            "[bold cyan]AlignTune Aligner Dashboard[/bold cyan]\n"
            "Starting interactive training inspector...\n"
        )

        self.start()

        try:
            # Keep dashboard running
            while self._running and self.aligner_session.is_training():
                time.sleep(1)

            # Training completed
            self.console.print(
                "[bold green]Training completed![/bold green]\n"
                "Final metrics:"
            )
            state = self.aligner_session.peek()
            for k, v in state.items():
                if isinstance(v, float):
                    self.console.print(f"  {k}: {v:.6f}")
                else:
                    self.console.print(f"  {k}: {v}")

        except KeyboardInterrupt:
            self.console.print("\n[bold yellow]Interrupted by user[/bold yellow]")
        finally:
            self.stop()
            logger.info("Dashboard exited")


def create_dashboard(aligner_session) -> Optional[AlignerDashboard]:
    """
    Create dashboard if rich is available.

    Args:
        aligner_session: AlignerSession instance

    Returns:
        AlignerDashboard or None if rich not available
    """
    if not RICH_AVAILABLE:
        logger.warning("rich not available - dashboard disabled")
        return None

    return AlignerDashboard(aligner_session)
