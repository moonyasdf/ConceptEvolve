# MEJORA: Módulo de visualización para la consola adaptado de ShinkaEvolve.

import logging
from typing import Optional, Callable, Any, List
import rich
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)

class DatabaseDisplay:
    """Maneja la visualización de resúmenes de la base de datos en la consola."""

    def __init__(
        self,
        cursor: sqlite3.Cursor,
        conn: sqlite3.Connection,
        config: Any,
        island_manager: Any,
        count_programs_func: Callable[[], int],
        get_best_program_func: Callable[[], Optional[Any]],
    ):
        self.cursor = cursor
        self.conn = conn
        self.config = config
        self.island_manager = island_manager
        self.count_programs_func = count_programs_func
        self.get_best_program_func = get_best_program_func

    def print_sampling_summary(
        self,
        parent: Any,
        archive_inspirations: List[Any],
        top_k_inspirations: List[Any],
        target_generation: int
    ):
        """Imprime un resumen del padre y las inspiraciones seleccionadas."""
        console = Console()
        
        table = Table(
            title=f"[bold red]Resumen de Muestreo - Generación {target_generation}[/bold red]",
            border_style="red",
            show_header=True,
            header_style="bold cyan",
            width=120,
        )

        table.add_column("Rol", style="cyan bold", width=12)
        table.add_column("Generación", style="magenta", justify="center", width=10)
        table.add_column("Isla", style="red", justify="center", width=8)
        table.add_column("Score", style="green", justify="right", width=10)
        table.add_column("Título", style="yellow", justify="left", width=70, overflow="ellipsis")
        
        def format_row(concept: Any, role: str):
            return [
                role,
                str(concept.generation),
                f"I-{concept.island_idx}" if concept.island_idx is not None else "N/A",
                f"{concept.combined_score:.2f}" if concept.combined_score is not None else "N/A",
                concept.title
            ]

        table.add_row(*format_row(parent, "[bold]PADRE[/bold]"))

        for i, concept in enumerate(archive_inspirations):
            table.add_row(*format_row(concept, f"Archivo-{i+1}"))
            
        for i, concept in enumerate(top_k_inspirations):
            table.add_row(*format_row(concept, f"TopK-{i+1}"))

        console.print(table)