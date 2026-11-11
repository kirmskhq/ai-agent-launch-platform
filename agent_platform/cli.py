"""The `agents` command line."""

import argparse
import json
import sys
from pathlib import Path

from agent_platform.settings import load_dotenv, paths


def _dirs(names: list[str]) -> list[Path]:
    from agent_platform.registry import discover

    return [paths().agents / n for n in names] if names else discover()


def _money(x: float, currency: str, digits: int = 0) -> str:
    return f"{x:,.{digits}f} {currency}"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.1%}"


def cmd_validate(args: argparse.Namespace) -> int:
    from agent_platform.registry import check_agent, check_readiness, proposals

    dirs = _dirs(args.agents)
    proposal_dirs = [] if args.agents else [paths().agents / r.name for r in proposals()]
    if not dirs and not proposal_dirs:
        print(f"no agents in {paths().agents}")
        return 0
    failed = 0
    for d in dirs + proposal_dirs:
        if not d.is_dir():
            problems = ["no such folder"]
        elif (d / "agent.yaml").is_file():
            problems = check_agent(d)
        else:
            problems = check_readiness(d)
        kind = "" if (d / "agent.yaml").is_file() else " (proposal)"
        print(f"{'FAIL' if problems else 'ok  '} {d.name}{kind}")
        for p in problems:
            print(f"     {p}")
        failed += bool(problems)
    return 1 if failed else 0


def print_readiness(r) -> None:
    print(f"{r.name}: {r.summary}")
    for problem in r.problems:
        print(f"     problem: {problem}")
    for c in r.checks:
        print(f"     {'pass' if c.passed else 'FAIL'}  {c.label}: {c.detail}")
    if r.reality is not None:
        print(
            f"     coverage {r.reality.coverage:.0%}, agreement {r.reality.agreement:.0%} "
            f"(the ceiling for the exam threshold), most autonomy allowed: {r.max_autonomy}"
        )
    for reason in r.reasons:
        print(f"     reason: {reason}")
    d = r.decision
    if d:
        print(f"     written decision: {d.outcome} by {d.by} on {d.decided_on}: {d.reason}")
    print(f"     cleared to build: {'yes' if r.cleared_to_build else 'no'}")


def cmd_readiness(args: argparse.Namespace) -> int:
    from agent_platform.readiness import PROPOSAL_FILE, assess

    root = paths().agents
    names = args.agents or sorted(
        p.name
        for p in (root.iterdir() if root.is_dir() else [])
        if (p / PROPOSAL_FILE).is_file() and not p.name.startswith((".", "_"))
    )
    if not names:
        print(f"no proposals in {root}")
        return 0
    failed = 0
    for name in names:
        r = assess(root / name)
        print_readiness(r)
        failed += bool(r.problems)
    return 1 if failed else 0


def cmd_propose(args: argparse.Namespace) -> int:
    import re
    import shutil

    from agent_platform.passport import NAME_PATTERN

    if not re.fullmatch(NAME_PATTERN, args.name):
        print("names use lowercase letters, digits and hyphens, 2 to 41 characters")
        return 1
    target = paths().agents / args.name
    if target.exists():
        print(f"{target} already exists")
        return 1
    starter = Path(__file__).parent / "starter"
    (target / "readiness").mkdir(parents=True)
    text = (starter / "proposal.yaml").read_text(encoding="utf-8")
    (target / "proposal.yaml").write_text(text.replace("NAME", args.name), encoding="utf-8")
    shutil.copy(starter / "reality_test.jsonl", target / "readiness" / "reality_test.jsonl")
    print(f"created {target}")
    print("next: fill in proposal.yaml, run the reality test with two staff members,")
    print(f"then check it: uv run agents readiness {args.name}")
    return 0


def print_report(r) -> None:
    c = r.counts
    label = "DEV" if r.case_set == "dev" else "PASS" if r.passed else "FAIL"
    print(
        f"{label} {r.agent} [{r.model}, {r.measured_on}, "
        f"{r.case_set} {r.cases_total} cases x {r.runs_per_case} runs]"
    )
    print(
        f"     answered correct {c['answered_correct']}, answered wrong {c['answered_wrong']}, "
        f"correct handover {c['correct_handover']}, unneeded handover "
        f"{c['unneeded_handover']}, infrastructure failure {c['infra_failure']}"
    )
    print(
        f"     answer accuracy {_pct(r.answer_accuracy)}, lower bound "
        f"{_pct(r.accuracy_lower_bound)} at {r.confidence:.0%} confidence "
        f"(need {_pct(r.min_accuracy)}, break-even {_pct(r.break_even_accuracy)})"
    )
    print(
        f"     settled by the agent {r.answered_share:.0%}, consistency {r.consistency:.0%}, "
        f"model cost per case {_money(r.cost_per_case, r.currency, 5)}, "
        f"p50 {r.latency_p50_s:.1f}s"
    )
    e, lb = r.effect, r.effect_at_lower_bound
    print(
        f"     monthly effect {_money(e['net'], r.currency)} "
        f"(at lower bound {_money(lb['net'], r.currency)}): staff saving "
        f"{_money(e['staff_saving'], r.currency)} - errors {_money(e['error_cost'], r.currency)}"
        f" - model {_money(e['model_cost'], r.currency)}"
    )
    for reason in r.reasons:
        print(f"     reason: {reason}")
    if r.measured_on == "mock":
        print("     note: measured on the mock model; this never makes an agent accepted")
    for case in r.cases:
        if not case.correct:
            run = case.runs[0]
            detail = run.error or json.dumps(run.output)[:160]
            note = "" if case.consistent else " (inconsistent)"
            print(f"     - {case.id} [{case.outcome}{note}] {detail}")


def cmd_exam(args: argparse.Namespace) -> int:
    from agent_platform.config import models_config
    from agent_platform.exam import ExamError, default_report_path, run_exam, save_report
    from agent_platform.gateway import has_key
    from agent_platform.passport import PassportError, load_passport
    from agent_platform.registry import record

    dirs = _dirs(args.agents)
    if not dirs:
        print(f"no agents in {paths().agents}")
        return 0
    failed = 0
    for d in dirs:
        try:
            passport = load_passport(d)
        except PassportError as e:
            print(f"FAIL {d.name}: passport invalid, no exam: {e}")
            failed += 1
            continue
        if passport.lifecycle == "retired" and not args.agents:
            continue
        if (
            args.reuse_accepted
            and args.set == "holdout"
            and args.model is None
            and args.runs is None
            and args.limit is None
        ):
            accepted = record(d)
            if accepted.runs:
                print(f"reusing accepted exam from {accepted.report.finished_at}")
                print_report(accepted.report)
                if args.out:
                    save_report(accepted.report, args.out)
                continue
        model = args.model or passport.model
        if args.skip_without_key:
            cfg = models_config()
            spec = cfg.models.get(model)
            if spec and not has_key(cfg.providers[spec.provider]):
                print(f"skip {d.name}: no key for {model}")
                continue
        try:
            report = run_exam(
                d,
                case_set=args.set,
                model=args.model,
                runs_per_case=args.runs,
                workers=args.workers,
                limit=args.limit,
            )
        except ExamError as e:
            print(f"FAIL {d.name}: {e}")
            failed += 1
            continue
        out = args.out or default_report_path(report, passport.model)
        save_report(report, out)
        print_report(report)
        print(f"     report: {out}")
        failed += not report.passed and args.set == "holdout"
    return 1 if failed else 0


def cmd_compare(args: argparse.Namespace) -> int:
    from agent_platform.compare import render
    from agent_platform.exam import load_report

    print(render(load_report(args.a), load_report(args.b)))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from agent_platform.registry import records

    items = records()
    if not items:
        print(f"no agents in {paths().agents}")
    for a in items:
        print(f"{a.status_label:<13} {a.name}")
        for reason in a.reasons:
            print(f"{'':<13}   {reason}")
    return 0


def cmd_invoke(args: argparse.Namespace) -> int:
    from agent_platform.runner import NotRunnable, invoke

    try:
        result = invoke(args.agent, json.loads(args.input))
    except (NotRunnable, LookupError) as e:
        print(f"FAIL {e}")
        return 1
    print(json.dumps(result.output, indent=2))
    fallback = ", fell back to a person" if result.fell_back else ""
    print(f"-- {result.model_calls} model calls, cost {result.cost:.6f}{fallback}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from agent_platform.runner import NotRunnable, run_job

    try:
        entry = run_job(args.agent, trigger="manual")
    except (NotRunnable, LookupError) as e:
        print(f"FAIL {e}")
        return 1
    print(json.dumps(entry, indent=2))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from agent_platform.config import models_config, platform_config
    from agent_platform.gateway import has_key

    cfg, platform = models_config(), platform_config()
    for alias, spec in cfg.models.items():
        provider = cfg.providers[spec.provider]
        price_in = spec.price_usd_per_1m.input * platform.usd_rate
        price_out = spec.price_usd_per_1m.output * platform.usd_rate
        key = "key set" if has_key(provider) else f"no key ({provider.api_key_env})"
        model_id = f"{spec.provider}/{spec.model}"
        print(
            f"{alias:<16} {model_id:<46} {price_in:>7.3f} / {price_out:>7.3f}"
            f" {platform.currency} per 1M tokens in/out  {key}"
        )
    return 0


def cmd_litellm_config(args: argparse.Namespace) -> int:
    from agent_platform import litellm_config

    if args.check:
        if litellm_config.is_current():
            print(f"{litellm_config.OUT} is up to date")
            return 0
        print(f"{litellm_config.OUT} is out of date: run `agents litellm-config`")
        return 1
    litellm_config.write()
    print(f"wrote {litellm_config.OUT}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"catalogue: http://{args.host}:{args.port}/  (Ctrl+C to stop)")
    uvicorn.run(
        "agent_platform.service:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def cmd_sandbox(args: argparse.Namespace) -> int:
    import uvicorn

    print(f"Harbourline Payments sandbox: http://{args.host}:{args.port}/docs  (Ctrl+C to stop)")
    uvicorn.run("sandbox.app:app", host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    from agent_platform.passport import json_schema

    print(json.dumps(json_schema(), indent=2))
    return 0


def cmd_accept_report(args: argparse.Namespace) -> int:
    from agent_platform.exam import acceptance_report_path, load_report, save_report
    from agent_platform.loader import fingerprint
    from agent_platform.readiness import assess
    from agent_platform.registry import derive_status, find

    report = load_report(args.report)
    agent = find(report.agent)
    if agent is None or agent.passport is None:
        print("FAIL report does not name a valid agent")
        return 1
    status, reasons = derive_status(
        agent.passport, report, fingerprint(agent.path, agent.passport), assess(agent.path)
    )
    if status != "running":
        print("FAIL " + "; ".join(reasons))
        return 1
    target = acceptance_report_path(agent.name)
    save_report(report, target)
    print(f"accepted {agent.name} on {report.model}")
    return 0


def cmd_value(args: argparse.Namespace) -> int:
    from agent_platform.production import measured

    print(json.dumps(measured(args.agent, weeks=args.weeks), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agents", description="Take AI agents from a department's process to production."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="Check passports, code and case files")
    p.add_argument("agents", nargs="*", help="Agent names; all agents if none given")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("propose", help="Start a proposal: proposal.yaml and a reality test")
    p.add_argument("name")
    p.set_defaults(func=cmd_propose)

    p = sub.add_parser("readiness", help="Check proposals before any agent code is written")
    p.add_argument("agents", nargs="*", help="Names; all proposals if none given")
    p.set_defaults(func=cmd_readiness)

    p = sub.add_parser("exam", help="Run the exam and write the report")
    p.add_argument("agents", nargs="*", help="Agent names; all agents if none given")
    p.add_argument("--set", choices=["holdout", "dev"], default="holdout")
    p.add_argument("--model", help="Run on another model (for comparison; never accepts)")
    p.add_argument("--runs", type=int, help="Runs per case (default: passport, 1 for dev)")
    p.add_argument("--limit", type=int, help="Only the first N cases")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, help="Where to write the report")
    p.add_argument(
        "--skip-without-key", action="store_true", help="Skip agents whose model has no key"
    )
    p.add_argument(
        "--reuse-accepted",
        action="store_true",
        help="Reuse a passed report only when its full code and configuration fingerprint matches",
    )
    p.set_defaults(func=cmd_exam)

    p = sub.add_parser("compare", help="Two exam reports side by side")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("status", help="What each agent is allowed to do, and why")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("invoke", help="Call a running agent once")
    p.add_argument("agent")
    p.add_argument("--input", required=True, help="Input as a JSON object")
    p.set_defaults(func=cmd_invoke)

    p = sub.add_parser("run", help="Run a running agent's scheduled job now")
    p.add_argument("agent")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("models", help="Models agents may use, with prices")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("litellm-config", help="Generate deploy/litellm.yaml from models.yaml")
    p.add_argument("--check", action="store_true", help="Fail if the file is out of date")
    p.set_defaults(func=cmd_litellm_config)

    p = sub.add_parser("serve", help="Start the service: catalogue, API, scheduler")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("sandbox", help="Start the fictional payments company's API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.set_defaults(func=cmd_sandbox)

    p = sub.add_parser("schema", help="Print the passport JSON Schema")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("accept-report", help="Install a matching passed real holdout report")
    p.add_argument("report", type=Path)
    p.set_defaults(func=cmd_accept_report)

    p = sub.add_parser("value", help="Measured effect and the value checkpoint")
    p.add_argument("agent")
    p.add_argument("--weeks", type=int, default=4)
    p.set_defaults(func=cmd_value)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
