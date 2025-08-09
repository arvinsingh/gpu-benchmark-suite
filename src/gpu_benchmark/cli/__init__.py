"""
Command-line interface for the GPU benchmark suite.
"""

from typing import List, Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from ..core.benchmark_runner import BenchmarkRunner
from ..core.device_info import DeviceInfo

app = typer.Typer(
    help="GPU Benchmark Suite - Performance testing for CUDA, Triton, and PyTorch"
)
console = Console()


@app.command()
def device_info(
    device_id: int = typer.Option(0, "--device", "-d", help="GPU device ID")
) -> None:
    """Display GPU device information."""
    try:
        device_info_obj = DeviceInfo()
        if device_info_obj.cuda_available:
            device_info_obj.print_device_info(device_id)
        else:
            rprint("[red]CUDA is not available on this system[/red]")
    except Exception as e:
        rprint(f"[red]Error getting device info: {e}[/red]")


@app.command()
def list_devices() -> None:
    """List all available GPU devices."""
    try:
        device_info = DeviceInfo()
        device_info.list_all_devices()
    except Exception as e:
        rprint(f"[red]Error listing devices: {e}[/red]")


@app.command()
def list_benchmarks(
    category: Optional[str] = typer.Option(
        None, "--category", "-c", help="Benchmark category to list"
    )
) -> None:
    """List available benchmarks."""
    try:
        runner = BenchmarkRunner()
        benchmarks = runner.list_benchmarks(category)

        if not benchmarks:
            rprint("[yellow]No benchmarks found[/yellow]")
            return

        table = Table(title="Available Benchmarks")
        table.add_column("Category", style="cyan")
        table.add_column("Benchmarks", style="green")

        for cat, bench_list in benchmarks.items():
            table.add_row(cat, ", ".join(bench_list))

        console.print(table)

    except Exception as e:
        rprint(f"[red]Error listing benchmarks: {e}[/red]")


@app.command()
def run(
    category: str = typer.Argument(..., help="Benchmark category to run"),
    benchmark: Optional[str] = typer.Option(
        None, "--benchmark", "-b", help="Specific benchmark to run"
    ),
    backends: List[str] = typer.Option(
        ["cuda", "triton", "pytorch"], "--backend", help="Backends to test"
    ),
    sizes: List[int] = typer.Option(
        [1024, 4096, 16384], "--size", help="Problem sizes to test"
    ),
    device_id: int = typer.Option(0, "--device", "-d", help="GPU device ID"),
    warmup: int = typer.Option(3, "--warmup", help="Number of warmup runs"),
    runs: int = typer.Option(10, "--runs", help="Number of benchmark runs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run benchmarks for a specific category or benchmark."""
    try:
        runner = BenchmarkRunner(device_id)

        if verbose:
            rprint(f"[blue]Running benchmarks on device {device_id}[/blue]")
            runner.get_device_info()

        # Run benchmarks
        if benchmark:
            # Run specific benchmark
            rprint(f"[green]Running benchmark: {category}/{benchmark}[/green]")
            results = runner.run_benchmark(
                category,
                benchmark,
                backends,
                sizes,
                warmup_runs=warmup,
                benchmark_runs=runs,
            )
        else:
            # Run all benchmarks in category
            rprint(f"[green]Running all benchmarks in category: {category}[/green]")
            results = runner.run_category(
                category, backends, sizes, warmup_runs=warmup, benchmark_runs=runs
            )

        if results:
            runner.print_results()

            # Show comparison
            comparison = runner.compare_results()
            if len(backends) > 1 and comparison:
                rprint("\n[cyan]Performance Comparison:[/cyan]")
                _print_comparison_table(comparison)
        else:
            rprint("[yellow]No results to display[/yellow]")

    except Exception as e:
        rprint(f"[red]Error running benchmarks: {e}[/red]")


@app.command()
def run_all(
    backends: List[str] = typer.Option(
        ["cuda", "triton", "pytorch"], "--backend", help="Backends to test"
    ),
    sizes: List[int] = typer.Option(
        [1024, 4096, 16384], "--size", help="Problem sizes to test"
    ),
    device_id: int = typer.Option(0, "--device", "-d", help="GPU device ID"),
    warmup: int = typer.Option(3, "--warmup", help="Number of warmup runs"),
    runs: int = typer.Option(10, "--runs", help="Number of benchmark runs"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run all available benchmarks."""
    try:
        runner = BenchmarkRunner(device_id)

        if verbose:
            rprint(f"[blue]Running all benchmarks on device {device_id}[/blue]")
            runner.get_device_info()

        rprint("[green]Running all benchmarks...[/green]")
        results = runner.run_all(
            backends, sizes, warmup_runs=warmup, benchmark_runs=runs
        )

        if results:
            runner.print_results()

            # show comparison
            comparison = runner.compare_results()
            if len(backends) > 1 and comparison:
                rprint("\n[cyan]Performance Comparison:[/cyan]")
                _print_comparison_table(comparison)
        else:
            rprint("[yellow]No results to display[/yellow]")

    except Exception as e:
        rprint(f"[red]Error running benchmarks: {e}[/red]")


@app.command()
def compare(
    benchmark: str = typer.Argument(
        ..., help="Benchmark to compare (format: category/name)"
    ),
    backends: List[str] = typer.Option(
        ["cuda", "triton", "pytorch"], "--backend", help="Backends to compare"
    ),
    sizes: List[int] = typer.Option(
        [1024, 4096, 16384], "--size", help="Problem sizes to test"
    ),
    device_id: int = typer.Option(0, "--device", "-d", help="GPU device ID"),
    baseline: str = typer.Option(
        "pytorch", "--baseline", help="Baseline backend for comparison"
    ),
) -> None:
    """Compare different implementations of a specific benchmark."""
    try:
        if "/" not in benchmark:
            rprint(
                "[red]Benchmark must be in format category/name (e.g., memory/vector-add)[/red]"
            )
            return

        category, name = benchmark.split("/", 1)

        runner = BenchmarkRunner(device_id)
        rprint(f"[green]Comparing {benchmark} across backends[/green]")

        results = runner.run_benchmark(category, name, backends, sizes)

        if results:
            runner.print_results()
            comparison = runner.compare_results(baseline)

            if comparison:
                rprint(f"\n[cyan]Speedup relative to {baseline}:[/cyan]")
                _print_comparison_table(comparison)
        else:
            rprint("[yellow]No results to display[/yellow]")

    except Exception as e:
        rprint(f"[red]Error comparing benchmarks: {e}[/red]")


@app.command()
def profile(
    benchmark: str = typer.Argument(
        ..., help="Benchmark to profile (format: category/name)"
    ),
    backend: str = typer.Argument(
        ..., help="Backend to profile (cuda, triton, pytorch)"
    ),
    size: int = typer.Option(4096, "--size", help="Problem size"),
    device_id: int = typer.Option(0, "--device", "-d", help="GPU device ID"),
) -> None:
    """Profile a specific benchmark implementation."""
    try:
        if "/" not in benchmark:
            rprint(
                "[red]Benchmark must be in format category/name (e.g., memory/vector-add)[/red]"
            )
            return

        category, name = benchmark.split("/", 1)

        runner = BenchmarkRunner(device_id)
        rprint(f"[green]Profiling {backend} implementation of {benchmark}[/green]")

        runner.profile_benchmark(category, name, backend, size)

    except Exception as e:
        rprint(f"[red]Error profiling benchmark: {e}[/red]")


def _print_comparison_table(comparison: dict) -> None:
    """Print a formatted comparison table."""
    table = Table(title="Backend Comparison")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Backend", style="green")
    table.add_column("Time (ms)", style="yellow")
    table.add_column("Speedup", style="magenta")
    table.add_column("GFLOPS", style="blue")
    table.add_column("GB/s", style="red")

    for bench_name, backends in comparison.items():
        for backend_name, metrics in backends.items():
            time_ms = (
                f"{metrics['execution_time']:.3f}"
                if metrics["execution_time"]
                else "N/A"
            )
            speedup = f"{metrics['speedup']:.2f}x" if metrics["speedup"] else "N/A"
            gflops = f"{metrics['flops']:.1f}" if metrics["flops"] else "N/A"
            bandwidth = f"{metrics['bandwidth']:.1f}" if metrics["bandwidth"] else "N/A"

            table.add_row(bench_name, backend_name, time_ms, speedup, gflops, bandwidth)

    console.print(table)


def main() -> None:
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
