"""MDK Orchestra unified CLI.

Commands:
    mdk-orchestra              Interactive wizard (Demo / API / Local / Quit)
    mdk-orchestra demo         End-to-end one-liner: simulator + Orchestra + dashboard
    mdk-orchestra simulator    Run only the simulator
    mdk-orchestra run          Full A/B run (Track A + Track B) with output capture
    mdk-orchestra train        Retrain the XGBoost predictors
    mdk-orchestra discover     Run pattern-discovery script
    mdk-orchestra --help

Every LLM-using command (demo, run) fails fast if no backend is reachable:
no `ANTHROPIC_API_KEY` set AND no compatible local server.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Environment probing
# ---------------------------------------------------------------------------


def _probe_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _probe_standard_local(host: str = "http://127.0.0.1:11434", timeout_s: float = 1.5) -> bool:
    """Return True if a standard-local LLM server answers at `host`."""
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout_s)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def _probe_ollama_models(host: str, timeout_s: float = 2.0) -> list[str]:
    """Return the list of model names Ollama reports at `host/api/tags`."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout_s) as resp:
            data = json.loads(resp.read())
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def _resolve_standard_local_host() -> str:
    """Read `config/llm_routing.yaml` to find the configured local host."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return "http://127.0.0.1:11434"
    cfg_path = _REPO_ROOT / "config" / "llm_routing.yaml"
    if not cfg_path.exists():
        return "http://127.0.0.1:11434"
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        return "http://127.0.0.1:11434"
    block = cfg.get("standard_local") or cfg.get("ollama") or {}
    return os.environ.get("MDK_LLM_STANDARD_LOCAL_HOST") or block.get(
        "host", "http://127.0.0.1:11434"
    )


def _require_llm_backend(profile: str) -> None:
    """Fail fast if the chosen profile has no reachable backend."""
    profile = profile or "full_api"
    has_anthropic = _probe_anthropic()
    local_host = _resolve_standard_local_host()
    has_local = _probe_standard_local(local_host)

    needs_anthropic = profile in {"full_api", "hybrid_economic", "opus_premium"}
    needs_local = profile in {"hybrid_economic", "full_local"}

    missing: list[str] = []
    if needs_anthropic and not has_anthropic:
        missing.append("anthropic")
    if needs_local and not has_local:
        missing.append(f"standard_local ({local_host})")

    if not missing:
        return

    sys.stderr.write(
        "\nError: No LLM backend available for profile '" + profile + "'.\n\n"
        "MDK Orchestra requires:\n"
    )
    if "anthropic" in missing:
        sys.stderr.write(
            "  - ANTHROPIC_API_KEY environment variable set (for API profiles)\n"
        )
    if any(m.startswith("standard_local") for m in missing):
        sys.stderr.write(
            f"  - A compatible local LLM server running at {local_host}\n"
            "    (Ollama / LM Studio / llama.cpp server / vLLM / ...)\n"
        )
    sys.stderr.write(
        "\nTo configure:\n"
        "  - API:   export ANTHROPIC_API_KEY=sk-ant-...\n"
        "  - Local: https://ollama.ai/download  then:  ollama pull qwen2.5:7b-instruct\n"
        "\n"
        "Then choose a profile:  --profile full_api | hybrid_economic | full_local\n"
    )
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Run directory helpers (session isolation)
# ---------------------------------------------------------------------------


def _create_run_dir(prefix: str = "", override: str | None = None) -> Path:
    """Create a fresh timestamped run directory and return it.

    Resolution order (via `shared.paths.get_runs_dir`):
      1. `override` argument (from --output-dir CLI flag)
      2. `MDK_RUNS_DIR` environment variable
      3. `~/.mdk-orchestra/runs/` (default, survives pipx uninstall)
    """
    from shared.paths import get_runs_dir
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{prefix}_{ts}" if prefix else ts
    run_dir = get_runs_dir(override) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "memory_snapshot_start").mkdir(exist_ok=True)
    return run_dir


def _snapshot_memory_files(target_dir: Path) -> None:
    """Copy current `agents/*_memory.md` into `target_dir`."""
    for f in (_REPO_ROOT / "agents").glob("*_memory.md"):
        shutil.copy(f, target_dir / f.name)


def _write_config_used(run_dir: Path, config: dict) -> None:
    """Persist the routing config snapshot for this run."""
    try:
        import yaml  # type: ignore[import-not-found]
        (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    except ImportError:
        (run_dir / "config_used.yaml").write_text(json.dumps(config, indent=2))


def _build_backend_summary(profile: str, custom_backend: dict | None = None) -> dict:
    """Construct a human-readable backend summary that the dashboard displays."""
    if custom_backend:
        return custom_backend
    # Default summaries for the three shipped profiles
    if profile == "full_api":
        return {
            "label": "Anthropic API",
            "detail": "Sonnet 4.6 (specialists) + Opus 4.7 (Maestro escalation)",
            "profile": profile,
        }
    if profile == "hybrid_economic":
        return {
            "label": "Hybrid",
            "detail": "Anthropic Maestro + local-LLM specialists",
            "profile": profile,
        }
    if profile == "full_local":
        host = _resolve_standard_local_host()
        return {
            "label": "Local",
            "detail": f"standard_local @ {host} · qwen2.5:7b-instruct",
            "profile": profile,
        }
    if profile == "opus_premium":
        return {
            "label": "Anthropic API (Opus premium)",
            "detail": "Opus 4.7 everywhere except specialists (Sonnet/Haiku)",
            "profile": profile,
        }
    return {"label": profile, "detail": "", "profile": profile}


# ---------------------------------------------------------------------------
# Wizard
# ---------------------------------------------------------------------------


_BANNER = r"""
┌─────────────────────────────────────────────────────┐
│  MDK Orchestra v0.1.0                               │
│  Predictive Maintenance for Bitcoin Mining          │
│                                                     │
│  multi-agent AI system                              │
└─────────────────────────────────────────────────────┘
"""


# ---------------------------------------------------------------------------
# Rich console (lazy: imported-when-used so legacy tests that patch stdout
# keep working on the plain `_prompt` path).
# ---------------------------------------------------------------------------


def _get_console():
    """Return a themed Rich Console (lazy import to keep cli.py loadable
    even if `rich` is absent in some degraded environment)."""
    try:
        from mdk_orchestra.cli_theme import get_console
        return get_console()
    except Exception:  # noqa: BLE001
        return None


def _render_banner(console) -> None:
    """Print the wizard banner as a Rich panel."""
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    content = Text.from_markup(
        "\n[brand]MDK Orchestra[/]  [dim]v0.1.0[/]\n"
        "[title]Predictive Maintenance for Bitcoin Mining[/]\n\n"
        "[subtitle]multi-agent AI system[/]\n",
        justify="center",
    )
    console.print(Panel(
        content,
        border_style="border.brand",
        box=box.ROUNDED,
        padding=(0, 4),
    ))


def _render_main_menu(console) -> None:
    """Render the 4-option main menu as a Rich table."""
    from rich.table import Table
    from rich import box
    t = Table(
        show_header=False,
        box=box.SIMPLE,
        padding=(0, 2),
        border_style="border.dim",
    )
    t.add_column(style="menu.num", width=4, justify="right")
    t.add_column(style="menu.name", width=12, no_wrap=True)
    t.add_column(style="menu.desc")
    t.add_row("[1]", "Demo", "Replay of real API run · 1× (10× simulated time)")
    t.add_row("[2]", "API", "Full system on Anthropic or OpenAI-compatible provider")
    t.add_row("[3]", "Local LLM", "Full system on Ollama or local inference server")
    t.add_row("[4]", "Explore", "Simulator + flag detection · no agents · no actions")
    console.print()
    console.print("  [title]How do you want to run?[/]")
    console.print(t)
    console.print("  [dim](Press Ctrl+C to exit)[/]")
    console.print()


def _prompt(msg: str, valid: set[str] | None = None, default: str | None = None) -> str:
    """Read a line from stdin, validating against `valid` (lowercased match)."""
    while True:
        suffix = f" [{default}]" if default else ""
        try:
            raw = input(f"{msg}{suffix}: ").strip()
        except EOFError:
            raise SystemExit(1)
        if not raw and default is not None:
            raw = default
        if valid is None or raw.lower() in {v.lower() for v in valid}:
            return raw
        print(f"  Invalid choice — must be one of {sorted(valid)}.")


def _render_demo_preflight(console) -> None:
    """Pre-flight Panel for Demo with full-API-run details."""
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    body = Table.grid(padding=(0, 2))
    body.add_column(style="muted", no_wrap=True)
    body.add_column(style="title")
    body.add_row("Flags consumed", "38")
    body.add_row("Orchestra decisions", "28")
    body.add_row("Autonomy ladder", "L1 · L2 · L3 · L4  [dim](full)[/]")
    body.add_row("", "")
    body.add_row("Replay speed", "[accent]1×[/]  [dim](real time, ~12 min)[/]")
    body.add_row("Cost", "[success]$0.00[/]")
    body.add_row("Requirements", "[dim]none (no API key, no LLM)[/]")
    console.print(Panel(
        body,
        title="[brand]Demo · Anthropic API run[/]",
        border_style="border.brand",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def _render_demo_summary(console, meta: dict, run_dir: Path) -> None:
    """Post-replay completion Panel."""
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    counts = meta.get("counts", {})
    body = Table.grid(padding=(0, 2))
    body.add_column(style="muted", no_wrap=True)
    body.add_column(style="title")
    body.add_row("Flags consumed", str(counts.get("flags", 0)))
    body.add_row("Orchestra decisions", str(counts.get("decisions", 0)))
    body.add_row("Autonomy distribution", "L1×13 · L2×6 · L3×5 · L4×4")
    body.add_row("", "")
    rel = run_dir.relative_to(_REPO_ROOT) if run_dir.is_relative_to(_REPO_ROOT) else run_dir
    body.add_row("Full event log", f"[dim]{rel}/decisions.jsonl[/]")
    console.print(Panel(
        body,
        title="[success]Demo Completed[/]",
        border_style="border.brand",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def _wizard_demo() -> int:
    console = _get_console()
    if console is not None:
        console.print()
        _render_demo_preflight(console)
    else:
        print("\n→ Demo mode selected.\n")
        print("Replay speed: 1× real-time (~12 min wall)")
        print()
    try:
        input("  Press Enter to start, or Ctrl+C to go back: ")
    except (KeyboardInterrupt, EOFError):
        print()
        return _run_wizard()

    from mdk_orchestra import replay as _replay

    source = _REPO_ROOT / "examples" / "demo_replay"
    if not source.exists():
        sys.stderr.write(
            f"\nError: demo replay assets missing at {source}\n"
            "Expected: examples/demo_replay/flags.jsonl etc.\n"
        )
        return 1

    # Prepare dirs + dashboard up front so we can wrap the blocking replay
    # in a live progress panel running on the main thread.
    run_dir, dashboard_proc = _replay.prepare_and_serve_empty(
        source=source, speed=1.0, run_id=None, dashboard_port=8000,
    )

    # Kick off the replay on a background thread — it writes to disk; the
    # ProgressWatcher reads those files in the main thread.
    import threading
    meta_holder: dict[str, object] = {}
    replay_err: dict[str, BaseException] = {}

    def _do_replay() -> None:
        try:
            writers = {
                stream: (run_dir / f"{stream}.jsonl").open("a", encoding="utf-8")
                for stream in _replay._REPLAY_STREAMS
            }
            try:
                meta = _replay._run_replay(source, run_dir, speed=1.0, open_files=writers)
                meta_holder["meta"] = meta
            finally:
                for w in writers.values():
                    try:
                        w.close()
                    except Exception:
                        pass
            (run_dir / "replay_meta.json").write_text(json.dumps(meta, indent=2))
        except BaseException as exc:
            replay_err["exc"] = exc

    t = threading.Thread(target=_do_replay, daemon=True)
    t.start()

    # Live progress panel for the duration of the replay.
    from mdk_orchestra.cli_progress import ProgressWatcher
    try:
        if console is not None:
            with ProgressWatcher(
                console, run_dir,
                title="Demo · Anthropic API run (1×)",
                dashboard_url="http://127.0.0.1:8000/",
                mode="demo",
                total_events=38 + 28 + 28 + 559,
            ):
                t.join()
        else:
            t.join()
    except KeyboardInterrupt:
        # Ctrl+C during demo → show partial report + return to main menu.
        _terminate_dashboard(dashboard_proc)
        if console is not None:
            console.print("\n  [warning]demo interrupted — partial report:[/]")
            _render_run_summary(console, run_dir, interrupted=True)
        return _run_wizard()

    if "exc" in replay_err:
        sys.stderr.write(f"\n[demo] replay error: {replay_err['exc']}\n")
        _terminate_dashboard(dashboard_proc)
        return _run_wizard()

    meta = meta_holder.get("meta") or {"counts": {}, "total_events": 0}

    print()  # spacer after the Live panel exits
    if console is not None:
        _render_demo_summary(console, meta, run_dir)
    else:
        print("═══ Demo completed ═══")
        counts = meta.get("counts", {})
        print(f"  Flags: {counts.get('flags', 0)}  Decisions: {counts.get('decisions', 0)}")

    time.sleep(1.0)

    while True:
        if console is not None:
            console.print()
            console.print("  [title]What next?[/]")
            console.print("    [menu.num][1][/]  Return to main menu")
            console.print("    [menu.num][2][/]  Show full log in terminal (pager)")
            console.print("    [menu.num][3][/]  Exit")
            console.print()
        else:
            print("\nWhat next?  [1] Return  [2] Pager  [3] Exit")
        choice = _prompt("  Select [1-3]", valid={"1", "2", "3"})
        if choice == "1":
            _terminate_dashboard(dashboard_proc)
            return _run_wizard()
        if choice == "2":
            log_path = run_dir / "decisions.jsonl"
            pager = os.environ.get("PAGER", "less")
            try:
                subprocess.call([pager, str(log_path)])
            except FileNotFoundError:
                print(log_path.read_text())
            print()
            continue
        # choice == "3"
        _terminate_dashboard(dashboard_proc)
        return 0


def _terminate_dashboard(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass


def _free_dashboard_port(port: int) -> None:
    """Kill any process bound to `port` so a fresh dashboard can bind.

    Without this, a stale dashboard from a crashed previous run keeps
    answering on localhost:<port>, pointed at its (old) stream dir —
    and the user sees "phantom" events from that prior session.
    """
    import signal as _signal
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return
    for raw in result.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            pid = int(raw)
        except ValueError:
            continue
        try:
            os.kill(pid, _signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    if result.stdout.strip():
        time.sleep(0.5)


def _wizard_api() -> int:
    while True:
        print("\n→ API mode selected.\n")
        print("Choose provider:")
        print("  [1]  Anthropic")
        print("       Models: Sonnet 4.6 (specialists) + Opus 4.7 (Maestro)")
        print("       Env var: ANTHROPIC_API_KEY")
        print()
        print("  [2]  Custom OpenAI-compatible")
        print("       For: OpenAI, Groq, Together, OpenRouter, DeepSeek, Mistral, ...")
        print("       You will specify: base_url, api_key env var, model")
        print()
        print("  [3]  Back")
        print()
        choice = _prompt("Select [1-3]", valid={"1", "2", "3"})
        if choice == "3":
            return _run_wizard()
        if choice == "1":
            return _wizard_api_anthropic()
        if choice == "2":
            return _wizard_api_custom()


def _wizard_api_anthropic() -> int:
    print()
    print("Configuration:")
    print("  Backend: Anthropic (Sonnet 4.6 + Opus 4.7)")
    print()
    if not _collect_api_key_inline("ANTHROPIC_API_KEY", "Anthropic"):
        return _run_wizard()
    duration = _prompt_duration_minutes()
    # Fault intensity is no longer a user choice — we use the same
    # balanced mix as the Demo replay so the flag-arrival rate feels
    # familiar (~3 flags / simulated-min on 50 miners).
    ns = argparse.Namespace(
        profile="full_api", duration=duration, miners=50, seed=42,
        fault_mix="balanced", run_id=None, output=None,
    )
    return _cmd_run(ns)


def _collect_api_key_inline(env_var: str, provider_label: str) -> bool:
    """Ask for an API key inline, or use the existing env var.

    Returns True if a key is available after this call; False otherwise
    (in which case a user-facing error has already been printed).
    """
    try:
        # Prompt split over two lines so long provider names don't get
        # truncated in narrow terminals. Empty input still falls back
        # to the env var, but we don't advertise it on-screen.
        print(f"Paste your {provider_label} API key below.")
        raw = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    if raw:
        os.environ[env_var] = raw
        suffix = raw[-4:] if len(raw) >= 4 else raw
        print(f"✓ Key accepted (ending ...{suffix})")
        return True
    existing = os.environ.get(env_var)
    if existing:
        suffix = existing[-4:] if len(existing) >= 4 else existing
        print(f"✓ Using {env_var} from env (ending ...{suffix})")
        return True
    sys.stderr.write(
        f"\n❌ No key provided and {env_var} not set.\n"
        f"   Run `export {env_var}=sk-...` or paste one here.\n\n"
    )
    return False


def _wizard_api_custom() -> int:
    print("\nProvide your provider details.")
    host = _prompt("Base URL (e.g. https://api.groq.com/openai)")
    api_key_env = _prompt("API key environment variable name (e.g. GROQ_API_KEY)")
    model = _prompt("Model name (e.g. llama-3.3-70b-versatile)")
    if not _collect_api_key_inline(api_key_env, api_key_env):
        return _run_wizard()
    print(f"\nConfiguration:")
    print(f"  Profile       :  full_api (customized)")
    print(f"  Backend       :  standard_api")
    print(f"  Base URL      :  {host}")
    print(f"  Model         :  {model}")
    print()
    duration = _prompt_duration_minutes()
    # Fault intensity removed from the UX — balanced mix matches demo.
    # Build a one-shot YAML that routes every slot through the custom provider
    custom_cfg = _build_custom_api_config(host, api_key_env, model)
    return _run_with_inline_config(
        custom_cfg,
        duration=duration,
        backend_summary={
            "label": "Remote API",
            "detail": f"{host} · {model}",
            "profile": "full_api (custom)",
        },
    )


def _wizard_local() -> int:
    print("\n→ Local LLM mode selected.\n")
    default_host = "http://127.0.0.1:11434"
    models = _probe_ollama_models(default_host)

    if models:
        print(f"Ollama detected at {default_host}. Available models:")
        for m in models:
            print(f"  - {m}")
        print()
        print("  [1]  Use Ollama at localhost with the first listed model")
        print("  [2]  Use Ollama at localhost with a different model (specify)")
        print("  [3]  Use different local server (LM Studio, llama.cpp, vLLM, custom)")
        print("  [4]  Back")
        print()
        choice = _prompt("Select [1-4]", valid={"1", "2", "3", "4"})
        if choice == "4":
            return _run_wizard()
        if choice == "1":
            return _run_local_mode(default_host, models[0])
        if choice == "2":
            model = _prompt("Model name (from the list above)", valid=set(models))
            return _run_local_mode(default_host, model)
        if choice == "3":
            return _wizard_local_custom()

    print(f"Ollama not detected at {default_host}.")
    print()
    print("  [1]  Specify Ollama host (remote server)")
    print("  [2]  Specify other local server (LM Studio, llama.cpp, vLLM, custom)")
    print("  [3]  Back")
    print()
    choice = _prompt("Select [1-3]", valid={"1", "2", "3"})
    if choice == "3":
        return _run_wizard()
    return _wizard_local_custom()


def _wizard_local_custom() -> int:
    host = _prompt("Host URL (e.g. http://192.168.1.75:11434)")
    model = _prompt("Model name served by host")
    if not _probe_standard_local(host):
        proceed = _prompt(f"Warning: could not reach {host}. Proceed anyway? [y/N]", default="n")
        if proceed.lower() != "y":
            return _wizard_local()
    return _run_local_mode(host, model)


def _run_local_mode(host: str, model: str) -> int:
    print(f"\nConfiguration:")
    print(f"  Profile       :  full_local")
    print(f"  Backend       :  standard_local")
    print(f"  Host          :  {host}")
    print(f"  Model         :  {model}")
    print()
    duration = _prompt_duration_minutes()
    # Fault intensity removed from UX — balanced mix matches demo.
    custom_cfg = _build_local_config(host, model)
    os.environ["MDK_LLM_STANDARD_LOCAL_HOST"] = host
    return _run_with_inline_config(
        custom_cfg,
        duration=duration,
        backend_summary={
            "label": "Local",
            "detail": f"{host} · {model}",
            "profile": "full_local",
        },
    )


def _prompt_duration_minutes() -> int:
    """Ask for an integer minute count of **simulated** time.

    The simulator runs at 10× wall speed, so `N` simulated minutes finish
    in `N × 60 / 10 = N × 6` seconds of real time — we state the
    conversion explicitly in the prompt so the user is never surprised.

    Tolerates trailing 'm', empty input (→ default 30), float input (→
    rounded), and garbage (→ default + warning). Returns int (the
    downstream `ab_experiment.main --duration-min` is int-typed).
    """
    # Clarified in two lines: what to type, what it means. The simulator
    # runs at 10× wall speed, so e.g. 20 simulated minutes = ~2 minutes
    # of real wall-clock waiting.
    print("Simulated duration in minutes (10× accelerated).")
    print("  e.g. type 20 for a ~2 min run that simulates 20 min.")
    raw = _prompt(
        "  Duration",
        default="30",
    ).strip().rstrip("m")
    try:
        value = int(raw)
    except ValueError:
        try:
            value = int(round(float(raw)))
        except ValueError:
            print(f"  Invalid duration — using 30 sim min (3 min wall).")
            return 30
    value = max(1, value)
    wall_s = value * 6
    wall_human = f"{wall_s // 60}m {wall_s % 60}s" if wall_s >= 60 else f"{wall_s}s"
    print(f"  → {value} sim min  (≈ {wall_human} wall-clock at 10× speed)")
    return value


def _prompt_fault_intensity() -> tuple[str, float]:
    """Ask how many faults to inject.

    Returns `(fault_mix, fault_rate)` where:
      - `fault_mix` is `"random"` or `"balanced"` (balanced = all 4 fault
        types round-robin; what the simulator actually supports today).
      - `fault_rate` is a scale factor that a future simulator knob may
        honour. Today's simulator doesn't read it directly — the choice
        maps to human expectations and logs it in config_used.yaml for
        traceability.
    """
    print()
    print("  Fault intensity:")
    print("    [1]  low      — a few faults, mostly nominal fleet")
    print("    [2]  medium   — balanced fault mix across 4 types (default)")
    print("    [3]  high     — many faults, lots of alerts to see Orchestra under pressure")
    print()
    choice = _prompt("  Select [1-3]", valid={"1", "2", "3"}, default="2")
    if choice == "1":
        return ("balanced", 0.5)
    if choice == "3":
        return ("balanced", 1.5)
    return ("balanced", 1.0)


def _build_custom_api_config(host: str, api_key_env: str, model: str) -> dict:
    """Construct an inline routing config that routes every slot to the custom provider."""
    slot = {"backend": "standard_api", "host": host, "api_key_env": api_key_env, "model": model}
    return {
        "standard_api": {"timeout_s": 120, "tool_call_retries": 2},
        "default": slot,
        "agents": {
            "maestro": {"dispatch": slot, "escalation": slot, "curation": slot},
            "specialists": {
                "voltage": slot, "hashrate": slot, "environment": slot, "power": slot,
            },
        },
    }


def _build_local_config(host: str, model: str) -> dict:
    """Construct an inline routing config for full_local on a custom host."""
    slot = {"backend": "standard_local", "model": model}
    return {
        "standard_local": {"host": host, "timeout_s": 120, "tool_call_retries": 2},
        "default": slot,
        "agents": {
            "maestro": {"dispatch": slot, "escalation": slot, "curation": slot},
            "specialists": {
                "voltage": slot, "hashrate": slot, "environment": slot, "power": slot,
            },
        },
    }


def _preload_demo_memory() -> None:
    """Copy sample memories into agents/, with a confirm prompt if non-empty."""
    examples = _REPO_ROOT / "examples" / "memories"
    targets = [
        ("sample_maestro_memory.md", "maestro_memory.md"),
        ("sample_hashrate_memory.md", "hashrate_memory.md"),
    ]
    # Check if any current memory is non-empty
    non_empty = []
    for _, tgt in targets:
        tgt_path = _REPO_ROOT / "agents" / tgt
        if tgt_path.exists() and "## Pattern:" in tgt_path.read_text():
            non_empty.append(tgt)
    if non_empty:
        print(f"\nExisting memory detected in: {', '.join(non_empty)}")
        confirm = _prompt("Overwrite with demo sample? [y/N]", default="n")
        if confirm.lower() != "y":
            print("  Keeping existing memory.")
            return
        backup = _REPO_ROOT / f".memory_backup_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup.mkdir(exist_ok=True)
        for _, tgt in targets:
            src = _REPO_ROOT / "agents" / tgt
            if src.exists():
                shutil.copy(src, backup / tgt)
        print(f"  Backed up to {backup.relative_to(_REPO_ROOT)}/")
    for src_name, tgt in targets:
        src = examples / src_name
        if src.exists():
            shutil.copy(src, _REPO_ROOT / "agents" / tgt)


def _run_wizard() -> int:
    console = _get_console()
    if console is not None:
        _render_banner(console)
        _render_main_menu(console)
    else:
        # Fallback for environments without rich: the old plain menu.
        sys.stdout.write(_BANNER)
        print("How do you want to run?\n")
        print("  [1]  Demo         Replay full API run · no API, no LLM")
        print("  [2]  API          Full system on remote LLM API")
        print("  [3]  Local LLM    Full system on local inference server")
        print("  [4]  Explore      Simulator + flag detection · no agents")
        print("\n(Press Ctrl+C to exit)\n")
    try:
        choice = _prompt("  Select [1-4]", valid={"1", "2", "3", "4"})
    except KeyboardInterrupt:
        print("\nGoodbye.")
        return 0
    if choice == "1":
        return _wizard_demo()
    if choice == "2":
        return _wizard_api()
    if choice == "3":
        return _wizard_local()
    if choice == "4":
        return _wizard_explore()
    return 0


_EXPLORE_AUTONOMY_BY_SEVERITY = {
    "info": ("observe", "L1_observe"),
    "warn": ("alert_operator", "L2_suggest"),
    "crit": ("throttle", "L3_bounded_auto"),
}

_EXPLORE_PRIMARY_BY_FLAG_TYPE = {
    "voltage_drift":                "voltage_agent",
    "hashrate_degradation":         "hashrate_agent",
    "chip_instability_precursor":   "hashrate_agent",
    "hashboard_failure_precursor":  "hashrate_agent",
    "thermal_runaway":              "environment_agent",
    "fan_anomaly":                  "environment_agent",
    "power_instability":            "power_agent",
    "chip_variance_high":           "voltage_agent",
    "anomaly_composite":            "hashrate_agent",
}


def _explore_simulate_decisions(run_dir: Path, stop: "threading.Event") -> None:
    """Tail `run_dir/flags.jsonl` and emit synthetic `orchestrator_decision`
    events for each flag ~10-14 s later.

    Used by Explore mode so the dashboard shows the flag → Orchestra →
    decision cycle even though the real agent layer isn't running. Each
    emitted decision carries a reasoning_trace prefixed with "Risolto
    (simulato)" so consumers can tell pedagogical output from a real
    Maestro trace.
    """
    import threading, random, uuid
    flags_path = run_dir / "flags.jsonl"
    decisions_path = run_dir / "decisions.jsonl"
    actions_path = run_dir / "actions.jsonl"

    # Wait for flags file to appear
    while not flags_path.exists() and not stop.is_set():
        stop.wait(0.3)

    seen: set[str] = set()
    pending: list[tuple[float, dict]] = []  # (fire_at_monotonic, flag_data)

    try:
        f = flags_path.open("r", encoding="utf-8")
    except OSError:
        return
    # Start from end so we only react to flags raised during this session
    f.seek(0, 2)

    while not stop.is_set():
        # 1) ingest new flags
        line = f.readline()
        while line:
            line = line.strip()
            if line:
                try:
                    env = json.loads(line)
                except json.JSONDecodeError:
                    line = f.readline()
                    continue
                fd = env.get("data") or {}
                fid = fd.get("flag_id")
                if fid and fid not in seen:
                    seen.add(fid)
                    delay = random.uniform(10.0, 14.0)
                    pending.append((time.monotonic() + delay, fd))
            line = f.readline()

        # 2) fire any due pending
        now = time.monotonic()
        still_pending: list[tuple[float, dict]] = []
        for fire_at, fd in pending:
            if now >= fire_at:
                _explore_emit_fake_decision(decisions_path, actions_path, fd)
            else:
                still_pending.append((fire_at, fd))
        pending = still_pending

        stop.wait(0.5)

    try:
        f.close()
    except Exception:
        pass


def _explore_emit_fake_decision(
    decisions_path: Path,
    actions_path: Path,
    flag_data: dict,
) -> None:
    """Write a synthetic orchestrator_decision + action_taken for a flag."""
    import uuid, random
    severity = flag_data.get("severity", "info")
    miner_id = flag_data.get("miner_id", "m000")
    flag_id = flag_data.get("flag_id", "flg_?")
    flag_type = flag_data.get("flag_type", "unknown")

    action, autonomy = _EXPLORE_AUTONOMY_BY_SEVERITY.get(
        severity, ("observe", "L1_observe")
    )
    primary = _EXPLORE_PRIMARY_BY_FLAG_TYPE.get(flag_type, "hashrate_agent")
    decision_id = f"dec_sim_{uuid.uuid4().hex[:10]}"
    ts = _dt.datetime.utcnow().isoformat(timespec="microseconds") + "Z"

    decision_envelope = {
        "event": "orchestrator_decision",
        "ts": ts,
        "source": "explore_sim",
        "data": {
            "decision_id": decision_id,
            "flag_id": flag_id,
            "miner_id": miner_id,
            "action": action,
            "action_params": {},
            "autonomy_level": autonomy,
            "confidence": round(random.uniform(0.78, 0.93), 2),
            "reasoning_trace": (
                f"Resolved (simulated — play Demo for a replay of a real run). "
                f"In a live Orchestra run, Maestro would dispatch this "
                f"{flag_type}/{severity} flag on {miner_id} to the "
                f"{primary}. This Explore session doesn't run the agent layer; "
                f"this decision is a UI placeholder."
            ),
            "consulted_agents": [primary],
            "total_cost_usd": 0.0,
            "total_latency_ms": 0.0,
            "pending_human_approval": False,
        },
    }
    action_envelope = {
        "event": "action_taken",
        "ts": ts,
        "source": "explore_sim",
        "data": {
            "action_id": f"act_sim_{uuid.uuid4().hex[:10]}",
            "decision_id": decision_id,
            "miner_id": miner_id,
            "action": action,
            "status": "simulated",
            "outcome_expected": "Simulated (Explore mode — no real executor)",
        },
    }
    try:
        with decisions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(decision_envelope) + "\n")
        with actions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(action_envelope) + "\n")
    except OSError:
        pass


def _render_explore_preflight(console) -> None:
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    body = Table.grid(padding=(0, 2))
    body.add_column(style="muted", no_wrap=True)
    body.add_column(style="title")
    body.add_row("Simulator", "50 miners · 10× accelerated · balanced fault mix")
    body.add_row("Detection", "Rule engine + 2 XGBoost predictors (active)")
    body.add_row("Dashboard", "Fleet telemetry + live flags")
    body.add_row("", "")
    body.add_row("No agents", "[dim]Maestro / specialists / executor: idle[/]")
    body.add_row("Duration", "[dim]open-ended (Ctrl+C to stop)[/]")
    console.print(Panel(
        body,
        title="[brand]Explore · Simulator + Flag Detection[/]",
        border_style="border.brand",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def _wizard_explore() -> int:
    """[4] Explore: simulator + detectors + dashboard; NO Orchestra/executor.

    Faults are injected and flagged so the operator can see the detection
    layer firing in real time, but nothing downstream (Maestro, specialists,
    executor) runs. `decisions.jsonl` and `actions.jsonl` stay empty for
    the whole session — they exist only because the dashboard expects them.
    """
    console = _get_console()
    if console is not None:
        console.print()
        _render_explore_preflight(console)
    else:
        print("\n→ Explore mode selected.\n")
        print("  Simulator + flag detection · no agents")
        print()
    try:
        input("  Press Enter to start: ")
    except (KeyboardInterrupt, EOFError):
        print()
        return _run_wizard()

    run_dir = _create_run_dir(prefix="explore")
    env = os.environ.copy()
    env["MDK_STREAM_DIR"] = str(run_dir)
    env["MDK_MEMORY_DIR"] = str(run_dir / "memory")
    (run_dir / "memory").mkdir(exist_ok=True)

    _write_backend_summary(run_dir, {
        "label": "Explore mode",
        "detail": "simulator + flag detection · no agents",
        "profile": "explore",
    })

    # Pre-create empty decision/action files. The simulator + detector will
    # write telemetry/kpis/snapshots/flags themselves.
    (run_dir / "decisions.jsonl").touch()
    (run_dir / "actions.jsonl").touch()

    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    def _spawn(name: str, args: list[str]) -> subprocess.Popen:
        return subprocess.Popen(
            [_PYTHON, "-m", name] + args,
            env=env, cwd=str(_REPO_ROOT),
            stdout=(log_dir / f"{name.replace('.', '_')}.log").open("w"),
            stderr=subprocess.STDOUT,
        )

    # Open-ended: simulator runs for 24 hours simulated (2.4h wall) — user
    # Ctrl+Cs to stop. Balanced fault mix so all four fault types surface.
    procs = [
        _spawn("simulator.main", [
            "--n-miners=50", "--seed=42",
            f"--duration={24 * 60 * 60}", "--speed=10",
            "--fault-mix=balanced",
            f"--output={run_dir / 'telemetry.jsonl'}",
        ]),
        _spawn("ingest.main", [
            f"--input-stream={run_dir / 'telemetry.jsonl'}",
            f"--kpi-output={run_dir / 'kpis.jsonl'}",
            f"--snap-output={run_dir / 'snapshots.jsonl'}",
        ]),
        _spawn("deterministic_tools.main", [
            f"--telemetry-stream={run_dir / 'telemetry.jsonl'}",
            f"--flag-output={run_dir / 'flags.jsonl'}",
            "--sensitivity=medium",
        ]),
    ]
    _free_dashboard_port(8000)
    procs.append(_spawn("dashboard.main", ["--port=8000"]))

    # Spawn a background "simulated Orchestra" thread that tails flags.jsonl
    # and writes synthetic decisions ~10-14 s later — so the user sees the
    # flag → Orchestra-working → decision cycle in Explore too. Each fake
    # decision carries a "Risolto (simulato)" reasoning trace so downstream
    # consumers know it's pedagogical, not a real Maestro output.
    import threading
    sim_stop = threading.Event()
    sim_thread = threading.Thread(
        target=_explore_simulate_decisions,
        args=(run_dir, sim_stop),
        name="explore-simulator",
        daemon=True,
    )
    sim_thread.start()

    from mdk_orchestra.cli_progress import ProgressWatcher
    try:
        if console is not None:
            with ProgressWatcher(
                console, run_dir,
                title="Explore · Simulator + Flag Detection",
                dashboard_url="http://127.0.0.1:8000/",
                mode="explore",
            ):
                procs[0].wait()
        else:
            sys.stderr.write(
                "\n[explore] dashboard: http://127.0.0.1:8000/\n"
                f"[explore] run dir: {run_dir}\n"
                "[explore] press Ctrl+C to stop.\n\n"
            )
            procs[0].wait()
    except KeyboardInterrupt:
        sys.stderr.write("\n[explore] stopping…\n")
        interrupted = True
    else:
        interrupted = False
    finally:
        sim_stop.set()
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
    # Show partial/full report and return to main menu (Ctrl+C never exits).
    _render_run_summary(console, run_dir, interrupted=interrupted)
    return _run_wizard()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _run_with_inline_config(
    config: dict,
    duration: float,
    backend_summary: dict,
) -> int:
    """Write the inline config to a temporary YAML, set MDK_LLM_PROFILE to
    empty (so the agents section is read), and dispatch to _cmd_run via env.
    """
    import tempfile
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        sys.stderr.write("Error: PyYAML required for custom configs\n")
        return 1

    run_dir = _create_run_dir(prefix="session")
    cfg_path = run_dir / "llm_routing_override.yaml"
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
    os.environ["MDK_LLM_CONFIG"] = str(cfg_path)  # Consumed by llm_backend if set
    # Note: current llm_backend reads _DEFAULT_CONFIG_PATH at module load. To
    # respect this override without modifying llm_backend, we write a shim:
    # point MDK_LLM_ORIGINAL_CONFIG env to tell downstream children. The
    # cleanest path is to pass the config content as an env var that a
    # small patch in llm_backend reads. For v0.1 we write the config to
    # the default path — the repo is a user's local install so this is a
    # transient, reversible copy.
    cfg_default = _REPO_ROOT / "config" / "llm_routing.yaml"
    cfg_backup = cfg_default.with_suffix(".yaml.wizard_backup")
    try:
        if cfg_default.exists() and not cfg_backup.exists():
            shutil.copy(cfg_default, cfg_backup)
        shutil.copy(cfg_path, cfg_default)

        _write_config_used(run_dir, config)
        _write_backend_summary(run_dir, backend_summary)

        # Dispatch to the generic run runner — no profile (config is explicit)
        ns = argparse.Namespace(
            profile=None, duration=int(duration), miners=50, seed=42,
            fault_mix="balanced", run_id=run_dir.name, output=str(run_dir.parent),
            run_dir_override=run_dir,
        )
        return _cmd_run(ns)
    finally:
        # Restore the original config
        if cfg_backup.exists():
            shutil.copy(cfg_backup, cfg_default)


def _write_backend_summary(run_dir: Path, summary: dict) -> None:
    """Persist the human-readable backend label the dashboard will render."""
    (run_dir / "backend_summary.json").write_text(json.dumps(summary, indent=2))


def _cmd_demo(args: argparse.Namespace) -> int:
    """One-liner end-to-end: simulator + Orchestra + dashboard."""
    _require_llm_backend(args.profile)
    if args.profile:
        os.environ["MDK_LLM_PROFILE"] = args.profile

    duration_min = args.duration
    run_dir = _create_run_dir(prefix="demo")
    stream_dir = run_dir
    memory_dir = run_dir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MDK_STREAM_DIR"] = str(stream_dir)
    env["MDK_MEMORY_DIR"] = str(memory_dir)

    # Persist config + backend badge so the dashboard can show it
    _snapshot_memory_files(run_dir / "memory_snapshot_start")
    _write_backend_summary(run_dir, _build_backend_summary(args.profile))
    _write_config_used(run_dir, {"profile": args.profile, "duration_min": duration_min,
                                  "miners": args.miners, "fault_mix": args.fault_mix,
                                  "seed": 42})

    procs = _launch_pipeline(
        stream_dir=stream_dir, env=env, duration_s=int(duration_min * 60),
        miners=args.miners, fault_mix=args.fault_mix,
        dashboard_port=args.dashboard_port, log_dir=run_dir / "logs",
    )

    sys.stderr.write(
        f"\n[demo] running {duration_min} min of simulated time "
        f"(~{int(duration_min * 60 // 10)} s wall + 45 s agent grace).\n"
        f"[demo] open http://127.0.0.1:{args.dashboard_port} to watch.\n"
        f"[demo] run dir: {run_dir}\n\n"
    )

    grace_s = 45
    wall_s = int(duration_min * 60) / 10.0 + grace_s
    try:
        time.sleep(wall_s)
    except KeyboardInterrupt:
        sys.stderr.write("\n[demo] interrupted — shutting down.\n")
    finally:
        _terminate(procs)
    sys.stderr.write(f"[demo] done. Run artifacts in {run_dir}\n")
    return 0


def _cmd_simulator(args: argparse.Namespace) -> int:
    """Run the simulator alone. Useful for dataset generation / offline ML."""
    cmd = [
        _PYTHON, "-m", "simulator.main",
        f"--n-miners={args.miners}",
        f"--seed={args.seed}",
        f"--duration={int(args.duration * 60)}",
        f"--speed={args.speed}",
        f"--fault-mix={args.fault_mix}",
        f"--output={args.output}",
    ]
    return subprocess.call(cmd, cwd=str(_REPO_ROOT))


def _cmd_run(args: argparse.Namespace) -> int:
    """Full A/B run via ab_experiment.main (Track A + Track B).

    Also launches the dashboard pointed at Track A's stream dir once that
    directory comes into existence. Subprocess stdout/stderr are captured
    to a launcher log so the wizard's Rich output stays clean.
    """
    if args.profile:
        _require_llm_backend(args.profile)
        os.environ["MDK_LLM_PROFILE"] = args.profile

    from shared.paths import get_runs_dir
    runs_root = get_runs_dir(args.output)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    launcher_log = runs_root / f"ab_run_launcher_{ts}.log"
    scenario = args.run_id or "default"
    dashboard_port = 8000

    cmd = [
        _PYTHON, "-m", "ab_experiment.main",
        f"--scenario={scenario}",
        f"--duration-min={args.duration}",
        f"--n-miners={args.miners}",
        f"--seed={args.seed}",
        f"--fault-mix={args.fault_mix}",
        "--api-mode",
        f"--output={args.output or runs_root}",
    ]

    console = _get_console()
    wall_s = args.duration * 6
    wall_human = (
        f"{wall_s // 60}m {wall_s % 60}s" if wall_s >= 60 else f"{wall_s}s"
    )
    if console is not None:
        from rich.panel import Panel
        from rich.text import Text
        from rich import box
        console.print()
        console.print(Panel(
            Text.from_markup(
                f"[title]Starting A/B run[/]\n\n"
                f"  [muted]Scenario[/]      {scenario}\n"
                f"  [muted]Duration[/]      {args.duration} sim min  "
                f"[dim](≈ {wall_human} wall at 10×)[/]\n"
                f"  [muted]Miners[/]        {args.miners}\n"
                f"  [muted]Profile[/]       {args.profile}\n"
                f"  [muted]Fault mix[/]     {args.fault_mix}\n"
                f"  [muted]Output dir[/]    {runs_root}\n"
                f"  [muted]Launcher log[/]  {launcher_log.name}\n\n"
                f"  [dim]Dashboard will open at http://127.0.0.1:{dashboard_port}/[/]\n"
                f"  [dim]Ctrl+C to stop.[/]"
            ),
            border_style="border.brand",
            box=box.ROUNDED,
            padding=(1, 2),
        ))
    else:
        sys.stderr.write(f"\n[run] launcher log: {launcher_log}\n\n")

    # Snapshot any existing run dirs for this scenario BEFORE launching
    # ab_experiment — the poll loop below needs to find only the NEW
    # directory, not a leftover from a previous invocation. Without this
    # snapshot the glob would match an older `<scenario>_<old_uuid>/` dir
    # that still exists on disk, and the dashboard would point at stale
    # data (user reported seeing flags from a previous API run).
    pre_existing_dirs = set(
        p for p in runs_root.glob(f"{scenario}_*") if p.is_dir()
    )

    # Launch ab_experiment in background so we can watch for the track_a
    # stream dir and bring up the dashboard mid-flight.
    logf = launcher_log.open("w")
    ab_proc = subprocess.Popen(
        cmd, cwd=str(_REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT,
    )

    # Poll for the NEW track_a stream dir — must be under a run-dir that
    # did NOT exist when we snapshotted pre_existing_dirs above.
    track_a_stream: Path | None = None
    for _ in range(60):
        if ab_proc.poll() is not None:
            break
        new_dirs = [
            p for p in runs_root.glob(f"{scenario}_*")
            if p.is_dir() and p not in pre_existing_dirs
        ]
        if new_dirs:
            # Pick the most recently created one (should only ever be 1)
            new_dirs.sort(key=lambda p: p.stat().st_ctime, reverse=True)
            candidate = new_dirs[0] / "track_a" / "stream"
            if candidate.exists():
                track_a_stream = candidate
                break
        time.sleep(0.5)

    dashboard_proc: subprocess.Popen | None = None
    if track_a_stream is not None:
        # Write the backend badge into the run dir so the dashboard header
        # shows the active profile (no runtime change otherwise).
        run_dir = track_a_stream.parent.parent
        _write_backend_summary(run_dir, _build_backend_summary(args.profile))
        # Mirror it into track_a/stream where the dashboard actually reads.
        _write_backend_summary(track_a_stream, _build_backend_summary(args.profile))

        # Ensure the logs dir exists before opening dashboard.log —
        # ab_experiment.main creates track_a/logs/ but the parent run_dir
        # doesn't have one by default.
        dash_log_dir = run_dir / "logs"
        dash_log_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["MDK_STREAM_DIR"] = str(track_a_stream)
        _free_dashboard_port(dashboard_port)
        dashboard_proc = subprocess.Popen(
            [_PYTHON, "-m", "dashboard.main", f"--port={dashboard_port}"],
            env=env, cwd=str(_REPO_ROOT),
            stdout=(dash_log_dir / "dashboard.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        time.sleep(1.5)
        if console is not None:
            console.print(
                f"  [accent]→ dashboard live at http://127.0.0.1:{dashboard_port}/[/]\n"
            )

    # Block until ab_experiment finishes (or user Ctrl+C). Wrap the wait
    # in a live Rich progress panel — same layout as Explore — so the
    # terminal mirrors what the browser dashboard shows.
    from mdk_orchestra.cli_progress import ProgressWatcher
    interrupted = False
    try:
        if console is not None and track_a_stream is not None:
            backend = _build_backend_summary(args.profile)
            title = f"A/B Run · {backend.get('label', args.profile)}"
            with ProgressWatcher(
                console, track_a_stream,
                title=title,
                dashboard_url=f"http://127.0.0.1:{dashboard_port}/",
                mode="live",
                show_cost=True,
            ):
                ab_proc.wait()
        else:
            ab_proc.wait()
    except KeyboardInterrupt:
        interrupted = True
        if console is not None:
            console.print("\n  [warning]interrupted — stopping run…[/]")
        ab_proc.terminate()
        try:
            ab_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ab_proc.kill()
    finally:
        if dashboard_proc is not None:
            _terminate_dashboard(dashboard_proc)
        logf.close()

    # Render a summary from whatever hit the stream files before exit.
    if track_a_stream is not None:
        _render_run_summary(console, track_a_stream, interrupted=interrupted)

    if interrupted:
        # Return to main menu instead of exiting — per UX policy: Ctrl+C
        # cancels the current operation and brings the user back to the
        # wizard root.
        return _run_wizard()
    return ab_proc.returncode or 0


def _render_run_summary(console, stream_dir: Path, *, interrupted: bool) -> None:
    """Build a compact post-run report from whatever landed in stream_dir."""
    if console is None:
        return

    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from collections import Counter

    def _count(fname: str) -> int:
        p = stream_dir / fname
        if not p.exists():
            return 0
        try:
            with p.open("rb") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    n_flags = _count("flags.jsonl")
    n_decisions = _count("decisions.jsonl")
    n_actions = _count("actions.jsonl")

    autonomy = Counter()
    total_cost = 0.0
    decisions_path = stream_dir / "decisions.jsonl"
    if decisions_path.exists():
        try:
            with decisions_path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    data = d.get("data") or {}
                    level = data.get("autonomy_level")
                    if level:
                        autonomy[level] += 1
                    cost = data.get("total_cost_usd")
                    if isinstance(cost, (int, float)):
                        total_cost += float(cost)
        except OSError:
            pass

    body = Table.grid(padding=(0, 2))
    body.add_column(style="muted", no_wrap=True)
    body.add_column(style="title")
    body.add_row("Flags raised", str(n_flags))
    body.add_row("Orchestra decisions", str(n_decisions))
    body.add_row("Actions taken", str(n_actions))
    if autonomy:
        parts = " · ".join(
            f"{lvl.split('_', 1)[0]}×{c}"
            for lvl, c in sorted(autonomy.items())
        )
        body.add_row("Autonomy distribution", parts)
    if total_cost > 0:
        body.add_row("Total API cost", f"${total_cost:.4f}")
    body.add_row("", "")
    run_dir = stream_dir.parent.parent if stream_dir.name == "stream" else stream_dir
    # Rich supports OSC 8 clickable links via markup; most modern terminals
    # (iTerm2, VS Code, macOS Terminal via hyperlinks.enable) honour file://
    # URLs and open the dir in Finder on click.
    link = f"file://{run_dir}"
    body.add_row(
        "Log files",
        f"[link={link}]{run_dir}[/link]",
    )

    title = "[warning]Run interrupted — partial report[/]" if interrupted else "[success]Run complete[/]"
    console.print()
    console.print(Panel(
        body,
        title=title,
        border_style="border.brand" if not interrupted else "border.dim",
        box=box.ROUNDED,
        padding=(1, 2),
    ))


def _cmd_train(args: argparse.Namespace) -> int:
    script = _REPO_ROOT / "scripts" / "retrain_xgb_miner_wise.py"
    if not script.exists():
        sys.stderr.write(f"Error: training script not found at {script}.\n")
        return 1
    cmd = [_PYTHON, str(script)]
    if args.data_dir:
        cmd.extend(["--data-dir", args.data_dir])
    return subprocess.call(cmd, cwd=str(_REPO_ROOT))


def _cmd_discover(args: argparse.Namespace) -> int:
    candidates = list((_REPO_ROOT / "scripts").glob("*pattern_discovery*.py"))
    if not candidates:
        sys.stderr.write("Error: no pattern_discovery script found under scripts/.\n")
        return 1
    cmd = [_PYTHON, str(candidates[0])]
    if args.hours:
        cmd.extend(["--hours", str(args.hours)])
    return subprocess.call(cmd, cwd=str(_REPO_ROOT))


def _cmd_replay(args: argparse.Namespace) -> int:
    """Replay a recorded run at configurable speed. Both `demo` and
    `replay` subcommands route here."""
    from mdk_orchestra import replay as _replay

    source = Path(args.source) if args.source else _REPO_ROOT / "examples" / "demo_replay"
    if not source.exists():
        sys.stderr.write(f"Error: source directory not found: {source}\n")
        return 1

    output_dir = getattr(args, "output_dir", None)

    if getattr(args, "no_dashboard", False):
        run_dir = _replay.replay_to_run(
            source, speed=args.speed, run_id=args.run_id, output_dir=output_dir,
        )
        print(f"Replay written to: {run_dir}")
        return 0

    run_dir, dashboard_proc = _replay.replay_and_serve(
        source,
        speed=args.speed,
        run_id=args.run_id,
        dashboard_port=args.dashboard_port,
        output_dir=output_dir,
    )
    sys.stderr.write(f"\n[replay] done. Run artifacts in {run_dir}\n")
    _terminate_dashboard(dashboard_proc)
    return 0


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _launch_pipeline(
    *, stream_dir: Path, env: dict, duration_s: int, miners: int,
    fault_mix: str, dashboard_port: int, log_dir: Path,
) -> list[subprocess.Popen]:
    """Launch simulator + ingest + detector + agents + action + dashboard."""
    log_dir.mkdir(parents=True, exist_ok=True)

    def _spawn(name: str, args: list[str]) -> subprocess.Popen:
        log_path = log_dir / f"{name.replace('.', '_')}.log"
        return subprocess.Popen(
            [_PYTHON, "-m", name] + args,
            env=env, cwd=str(_REPO_ROOT),
            stdout=log_path.open("w"), stderr=subprocess.STDOUT,
        )

    procs: list[subprocess.Popen] = []
    procs.append(_spawn("simulator.main", [
        f"--n-miners={miners}", "--seed=42",
        f"--duration={duration_s}", "--speed=10",
        f"--fault-mix={fault_mix}",
        f"--output={stream_dir / 'telemetry.jsonl'}",
    ]))
    procs.append(_spawn("ingest.main", [
        f"--input-stream={stream_dir / 'telemetry.jsonl'}",
        f"--kpi-output={stream_dir / 'kpis.jsonl'}",
        f"--snap-output={stream_dir / 'snapshots.jsonl'}",
    ]))
    procs.append(_spawn("deterministic_tools.main", [
        f"--telemetry-stream={stream_dir / 'telemetry.jsonl'}",
        f"--flag-output={stream_dir / 'flags.jsonl'}",
        "--sensitivity=medium",
    ]))
    procs.append(_spawn("agents.main", [
        f"--flag-stream={stream_dir / 'flags.jsonl'}", "--from-start",
    ]))
    procs.append(_spawn("action.main", [
        f"--decision-stream={stream_dir / 'decisions.jsonl'}", "--from-start",
    ]))
    procs.append(_spawn("dashboard.main", [f"--port={dashboard_port}"]))
    return procs


def _terminate(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except OSError:
                pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mdk-orchestra",
        description=(
            "MDK Orchestra — predictive maintenance for Bitcoin mining via "
            "multi-agent AI.\n\n"
            "Run `mdk-orchestra` without arguments to open the interactive wizard."
        ),
        epilog=(
            "Examples:\n"
            "  mdk-orchestra                               # interactive wizard\n"
            "  mdk-orchestra demo                          # replay full API run (1× real-time)\n"
            "  mdk-orchestra replay --speed 4              # faster replay\n"
            "  mdk-orchestra run --duration 60             # live 60-min run (needs API/LLM)\n"
            "  mdk-orchestra simulator --duration 120      # simulator only\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=False)

    demo = sub.add_parser(
        "demo",
        help="Replay full API run in real-time (no API/LLM needed)",
    )
    demo.add_argument("--speed", type=float, default=1.0,
                      help="Replay speed multiplier (default 1.0 = real-time)")
    demo.add_argument("--source", default=None,
                      help="Source dir (default: examples/demo_replay/)")
    demo.add_argument("--run-id", default=None, help="Run identifier")
    demo.add_argument("--dashboard-port", type=int, default=8000)
    demo.add_argument("--output-dir", default=None,
                      help="Override runs directory "
                           "(default: $MDK_RUNS_DIR or ~/.mdk-orchestra/runs/)")
    demo.set_defaults(func=_cmd_replay)

    replay = sub.add_parser(
        "replay",
        help="Alias for demo with configurable speed + source",
    )
    replay.add_argument("--speed", type=float, default=1.0)
    replay.add_argument("--source", default=None)
    replay.add_argument("--run-id", default=None)
    replay.add_argument("--dashboard-port", type=int, default=8000)
    replay.add_argument("--no-dashboard", action="store_true",
                        help="Write replayed stream to disk without launching the dashboard")
    replay.add_argument("--output-dir", default=None,
                        help="Override runs directory "
                             "(default: $MDK_RUNS_DIR or ~/.mdk-orchestra/runs/)")
    replay.set_defaults(func=_cmd_replay)

    sim = sub.add_parser("simulator", help="Run the simulator alone")
    sim.add_argument("--duration", type=float, default=60.0, help="Simulated minutes")
    sim.add_argument("--miners", type=int, default=50, help="Number of miners")
    sim.add_argument("--seed", type=int, default=42)
    sim.add_argument("--speed", type=float, default=10.0)
    sim.add_argument("--fault-mix", default="balanced", choices=["random", "balanced"])
    sim.add_argument("--output", default="runs/telemetry.jsonl")
    sim.set_defaults(func=_cmd_simulator)

    run = sub.add_parser("run", help="Full A/B run (Track A + Track B)")
    run.add_argument("--profile", default="full_api",
                     choices=["full_api", "hybrid_economic", "full_local", "opus_premium"])
    run.add_argument("--duration", type=int, default=60, help="Simulated minutes")
    run.add_argument("--miners", type=int, default=50)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--fault-mix", default="balanced", choices=["random", "balanced"])
    run.add_argument("--run-id", default=None)
    run.add_argument("--output", default=None)
    run.set_defaults(func=_cmd_run)

    train = sub.add_parser("train", help="Retrain the XGBoost predictors")
    train.add_argument("--data-dir", default=None)
    train.set_defaults(func=_cmd_train)

    disc = sub.add_parser("discover", help="Run pattern discovery on simulator data")
    disc.add_argument("--hours", type=float, default=None)
    disc.set_defaults(func=_cmd_discover)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # No subcommand → launch wizard
    try:
        if not getattr(args, "command", None):
            return _run_wizard()
        return int(args.func(args))
    except KeyboardInterrupt:
        # Graceful exit on Ctrl+C from anywhere — wizard, replay, or
        # any explicit subcommand that hasn't already handled the signal.
        sys.stdout.write("\nGoodbye.\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
