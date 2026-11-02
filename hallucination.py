import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

import analysis

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Set from --api-retries in main().
API_RETRIES = 8
BACKOFF_START_SECONDS = 5.0
BACKOFF_MAX_SECONDS = 120.0
RETRYABLE_API_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,  # includes timeouts
    openai.InternalServerError,
)


DEFAULT_DATA_DIR = "data/v3"


def load_data(data_dir):
    """Load the source packet, cases, prompts, and judge instructions.

    Expected layout:
        source.txt            the source packet (optional; omit for closed-book
                              tests, where prompts should not use {source})
        cases.json            list of {id, category, question, ground_truth}
        judge.txt             system prompt for the judge
        prompts/<variant>.txt prompt templates using {source} and {question}
    """
    root = Path(data_dir)

    def read(path):
        return path.read_text(encoding="utf-8").strip()

    cases = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    required = {"id", "category", "question", "ground_truth"}
    for case in cases:
        if set(case) != required:
            raise ValueError(f"Case {case.get('id')} must have fields {required}.")
    ids = [case["id"] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Case ids in cases.json must be unique.")

    prompts = {p.stem: read(p) for p in sorted((root / "prompts").glob("*.txt"))}
    if not prompts:
        raise ValueError(f"No prompt templates found in {root / 'prompts'}.")

    source_path = root / "source.txt"
    return {
        "source": read(source_path) if source_path.exists() else "",
        "cases": cases,
        "prompts": prompts,
        "judge_instructions": read(root / "judge.txt"),
    }


class EmptyAnswerError(RuntimeError):
    def __init__(self, details):
        super().__init__("Model returned an empty answer.")
        self.details = details


def create_with_backoff(client, **kwargs):
    """chat.completions.create, waiting out rate limits and server errors.

    A 429 returns instantly, so without waiting a rate-limited run would
    burn through its remaining questions as errors.
    """
    for attempt in range(API_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RETRYABLE_API_ERRORS as exc:
            if attempt == API_RETRIES:
                raise
            delay = min(BACKOFF_MAX_SECONDS, BACKOFF_START_SECONDS * 2 ** attempt)
            headers = getattr(getattr(exc, "response", None), "headers", None)
            try:
                asked = float(headers.get("retry-after")) if headers else 0.0
                delay = max(delay, min(asked, BACKOFF_MAX_SECONDS))
            except (TypeError, ValueError):
                pass
            delay *= random.uniform(0.8, 1.2)
            print(
                f"  {type(exc).__name__}; waiting {delay:.0f}s "
                f"(retry {attempt + 1}/{API_RETRIES})",
                flush=True,
            )
            time.sleep(delay)


REASONING_CHOICES = ("default", "off", "on", "low", "medium", "high")


def reasoning_param(mode):
    """OpenRouter `reasoning` request object for a --reasoning choice.

    "default" returns None: nothing is sent and the model/provider decides.
    """
    if mode in (None, "default"):
        return None
    if mode == "off":
        return {"enabled": False}
    if mode == "on":
        return {"enabled": True}
    return {"effort": mode}


def request_options(routing, reasoning_mode):
    """Merge provider routing and reasoning settings into one extra_body."""
    body = dict(routing or {})
    reasoning = reasoning_param(reasoning_mode)
    if reasoning:
        body["reasoning"] = reasoning
    return body or None


def complete_once(client, model, messages, temperature, max_tokens=None,
                  extra_body=None, meta=None):
    kwargs = {"max_tokens": max_tokens} if max_tokens else {}
    if extra_body:
        kwargs["extra_body"] = extra_body
    response = create_with_backoff(
        client,
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )
    provider = getattr(response, "provider", None)
    provider = str(provider) if provider is not None else None
    if meta is not None:
        meta["provider"] = provider
    # OpenRouter can return HTTP 200 with an error body and no choices.
    if not response.choices:
        raise RuntimeError(
            f"No choices returned: {getattr(response, 'error', None)}"
        )
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(choice.message, "reasoning", None)
    if meta is not None:
        meta["completion_tokens"] = getattr(usage, "completion_tokens", None)
        meta["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
        meta["reasoning"] = (
            reasoning if isinstance(reasoning, str) and reasoning else None
        )
    if choice.finish_reason not in ("stop", None):
        raise RuntimeError(
            f"Nonstandard finish reason: {choice.finish_reason}"
        )
    text = choice.message.content
    if not text or not text.strip():
        raise EmptyAnswerError({
            "finish_reason": choice.finish_reason,
            "content": text,
            "refusal": getattr(choice.message, "refusal", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            "provider": provider,
        })
    return text.strip()


def complete(client, model, messages, temperature, max_tokens=None,
             empty_retries=0, empty_log=None, extra_body=None, meta=None):
    """complete_once, retrying empty responses.

    Every call (both prompt variants and the judge) gets the same number of
    retries, and each empty response is appended to empty_log if given.
    """
    for attempt in range(empty_retries + 1):
        try:
            return complete_once(
                client, model, messages, temperature, max_tokens,
                extra_body, meta,
            )
        except EmptyAnswerError as exc:
            if empty_log is not None:
                empty_log.append(exc.details)
            if attempt == empty_retries:
                raise


def parse_grade(text):
    # Tolerate Markdown fences or stray text around the JSON object.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("No JSON object found in judge output.")

    grade = json.loads(text[start:end + 1])
    required = {
        "pass", "unsupported_claim", "unnecessary_abstention", "reason"
    }
    if not isinstance(grade, dict) or set(grade) != required:
        raise ValueError("Judge returned the wrong JSON fields.")
    for key in required - {"reason"}:
        if type(grade[key]) is not bool:
            raise ValueError(f"{key} must be a JSON boolean.")
    if not isinstance(grade["reason"], str):
        raise ValueError("reason must be a string.")
    if grade["pass"] and (
        grade["unsupported_claim"] or grade["unnecessary_abstention"]
    ):
        raise ValueError("Judge returned contradictory scores.")
    return grade


def evaluate(client, model, temperature, max_tokens, data, case, answer,
             empty_retries=0, extra_body=None):
    fields = {
        "category": case["category"],
        "question": case["question"],
        "ground_truth": case["ground_truth"],
        "candidate_answer": answer,
    }
    if data["source"]:
        fields = {"source_packet": data["source"], **fields}
    payload = json.dumps(fields, ensure_ascii=False)
    messages = [
        {"role": "system", "content": data["judge_instructions"]},
        {"role": "user", "content": payload},
    ]
    last_error = None
    raw = None
    for _ in range(3):
        raw = complete(
            client, model, messages, temperature, max_tokens, empty_retries,
            extra_body=extra_body,
        )
        try:
            return parse_grade(raw), raw
        except (ValueError, TypeError) as exc:
            last_error = str(exc)
            messages.extend([
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your grade could not be validated: "
                        + last_error
                        + ". Return only the required JSON object "
                          "with consistent boolean scores."
                    ),
                },
            ])
    raise ValueError(
        f"Judge failed validation: {last_error}. Last output: {raw}"
    )


def aggregate_grades(grades):
    """Majority vote over several judge passes (needs an odd count)."""
    if len(grades) == 1:
        return grades[0]
    need = len(grades) / 2
    unsupported = sum(g["unsupported_claim"] for g in grades) > need
    abstained = sum(g["unnecessary_abstention"] for g in grades) > need
    passed = (
        sum(g["pass"] for g in grades) > need
        and not unsupported and not abstained
    )
    reason = next((g for g in grades if g["pass"] == passed), grades[0])
    return {
        "pass": passed,
        "unsupported_claim": unsupported,
        "unnecessary_abstention": abstained,
        "reason": reason["reason"],
    }


def grade_record(client, args, judge_model, data, case, record):
    runs = []
    judge_body = request_options(None, args.judge_reasoning)
    for _ in range(args.judge_passes):
        grade, raw = evaluate(
            client, judge_model, args.judge_temperature,
            args.judge_max_tokens, data, case, record["answer"],
            args.empty_retries, judge_body,
        )
        runs.append({"grade": grade, "raw": raw})
    record["judge_runs"] = runs
    record["judge_raw"] = runs[0]["raw"]
    record["grade"] = aggregate_grades([run["grade"] for run in runs])


def record_error(record, exc):
    record["error"] = f"{type(exc).__name__}: {exc}"
    if record["answer"] is not None:
        record["error_type"] = "judge_error"
    elif isinstance(exc, EmptyAnswerError):
        record["error_type"] = "empty_answer"
    else:
        record["error_type"] = "api_error"


def new_record(trial, case, variant):
    return {
        "trial": trial,
        "question_id": case["id"],
        "category": case["category"],
        "variant": variant,
        "question": case["question"],
        "ground_truth": case["ground_truth"],
        "answer": None,
        "provider": None,
        "completion_tokens": None,
        "reasoning_tokens": None,
        "reasoning": None,
        "empty_responses": [],
        "grade": None,
        "judge_raw": None,
        "judge_runs": None,
        "error": None,
        "error_type": None,
    }


def report_progress(number, total, record):
    status = (
        "ERROR" if record["error"]
        else "PASS" if record["grade"]["pass"]
        else "FAIL"
    )
    reasoning = (
        f" (reasoning_tokens={record['reasoning_tokens']})"
        if record.get("reasoning_tokens") is not None else ""
    )
    print(
        f"[{number}/{total}] trial={record['trial']} "
        f"{record['question_id']} {record['variant']}: {status}{reasoning}",
        flush=True,
    )
    for empty in record["empty_responses"]:
        print(
            f"  empty response (finish_reason={empty['finish_reason']}, "
            f"completion_tokens={empty['completion_tokens']}, "
            f"provider={empty['provider']})",
            flush=True,
        )
    if record["error"]:
        print(f"  {record['error']}", flush=True)


def provider_routing(args):
    """OpenRouter provider preferences for the candidate model only."""
    routing = {}
    if args.provider_ignore:
        routing["ignore"] = [
            name.strip() for name in args.provider_ignore.split(",")
        ]
    if args.provider_only:
        routing["order"] = [
            name.strip() for name in args.provider_only.split(",")
        ]
        routing["allow_fallbacks"] = False
    return {"provider": routing} if routing else None


def generate(client, args, judge_model, data, out):
    cases, prompts = data["cases"], data["prompts"]
    extra_body = request_options(provider_routing(args), args.reasoning)
    jobs = [
        (trial, index, variant)
        for trial in range(1, args.trials + 1)
        for index in range(len(cases))
        for variant in prompts
    ]
    random.Random(args.order_seed).shuffle(jobs)
    records = []

    with (out / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for number, (trial, index, variant) in enumerate(jobs, 1):
            case = cases[index]
            record = new_record(trial, case, variant)
            meta = {}
            try:
                prompt = prompts[variant].format(
                    source=data["source"], question=case["question"]
                )
                # Each answer is generated in a fresh conversation.
                record["answer"] = complete(
                    client, args.model,
                    [{"role": "user", "content": prompt}],
                    args.temperature, args.max_tokens,
                    args.empty_retries, record["empty_responses"],
                    extra_body, meta,
                )
                grade_record(client, args, judge_model, data, case, record)
            except Exception as exc:
                # Errors are kept separate from incorrect answers.
                record_error(record, exc)
            record["provider"] = meta.get("provider")
            record["completion_tokens"] = meta.get("completion_tokens")
            record["reasoning_tokens"] = meta.get("reasoning_tokens")
            record["reasoning"] = meta.get("reasoning")

            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            report_progress(number, len(jobs), record)
    return records


def regrade(client, args, parser, judge_model, data, out, old_meta):
    """Re-judge the saved answers of an earlier run; no new generation."""
    source_dir = Path(args.regrade)
    old_records = [
        json.loads(line)
        for line in (source_dir / "answers.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    cases = {case["id"]: case for case in data["cases"]}
    for record in old_records:
        case = cases.get(record["question_id"])
        if case is None or case["question"] != record["question"]:
            parser.error(
                f"Question {record['question_id']} in {source_dir} does not "
                f"match {args.data_dir}/cases.json. Pass the --data-dir the "
                "run was made with (ground truth may change; questions "
                "may not)."
            )
    if data["source"] != old_meta["source"].strip():
        parser.error(
            f"{args.data_dir}/source.txt differs from the packet the "
            "answers were generated from."
        )

    records = []
    with (out / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for number, record in enumerate(old_records, 1):
            case = cases[record["question_id"]]
            record.setdefault("empty_responses", [])
            record["ground_truth"] = case["ground_truth"]
            if record["answer"] is not None:
                record.update(
                    grade=None, judge_raw=None, judge_runs=None,
                    error=None, error_type=None,
                )
                try:
                    grade_record(client, args, judge_model, data, case, record)
                except Exception as exc:
                    record_error(record, exc)
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            report_progress(number, len(old_records), record)
    return records


def main():
    global API_RETRIES
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="OpenRouter model slug, e.g. google/gemma-3-27b-it "
             "(required unless --regrade is used)",
    )
    parser.add_argument(
        "--regrade", metavar="RUN_DIR",
        help="re-judge the saved answers in an earlier results/<run_id> "
             "folder instead of generating new ones; writes a new run "
             "folder",
    )
    parser.add_argument(
        "--data-dir",
        help="folder with source.txt, cases.json, judge.txt, and prompts/ "
             f"(default {DEFAULT_DATA_DIR}; with --regrade, the folder the "
             "run recorded)",
    )
    parser.add_argument("--base-url", default=OPENROUTER_BASE_URL)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--empty-retries", type=int, default=3,
        help="times to retry an empty model response, for candidate and "
             "judge calls alike (default 3)",
    )
    parser.add_argument(
        "--api-retries", type=int, default=API_RETRIES,
        help="times to retry a rate-limited (429), timed-out, or 5xx API "
             "call, waiting 5s, 10s, 20s... up to 2 minutes between tries "
             f"(default {API_RETRIES})",
    )
    parser.add_argument(
        "--provider-ignore", metavar="NAMES",
        help="comma-separated OpenRouter providers the candidate model must "
             "not use, e.g. Parasail",
    )
    parser.add_argument(
        "--provider-only", metavar="NAMES",
        help="comma-separated OpenRouter providers to use for the candidate "
             "model, with no fallback to others, e.g. DeepInfra",
    )
    parser.add_argument(
        "--reasoning", choices=REASONING_CHOICES, default="default",
        help="reasoning setting sent for the candidate model: default sends "
             "nothing, off disables it, on enables it, low/medium/high set "
             "the effort. Reasoning tokens and text are saved per answer",
    )
    parser.add_argument(
        "--judge-reasoning", choices=REASONING_CHOICES, default="default",
        help="the same setting for the judge model",
    )
    parser.add_argument(
        "--judge-model",
        help="OpenRouter slug for the judge (defaults to --model; a "
             "different, stronger model is recommended)",
    )
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-max-tokens", type=int)
    parser.add_argument(
        "--judge-passes", type=int, default=1,
        help="grade each answer this many times and take the majority; "
             "must be odd. More than 1 also measures judge consistency",
    )
    parser.add_argument("--order-seed", type=int, default=42)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    if not args.regrade and not args.model:
        parser.error("--model is required unless --regrade is used")
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.empty_retries < 0 or args.api_retries < 0:
        parser.error("--empty-retries and --api-retries cannot be negative")
    API_RETRIES = args.api_retries
    if args.judge_passes < 1 or args.judge_passes % 2 == 0:
        parser.error("--judge-passes must be an odd number")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        parser.error("Set OPENROUTER_API_KEY in your environment or .env file.")

    old_meta = None
    if args.regrade:
        old_meta = json.loads(
            (Path(args.regrade) / "metadata.json").read_text(encoding="utf-8")
        )
        judge_model = (
            args.judge_model
            or old_meta.get("judge_model_effective")
            or old_meta["model"]
        )
    else:
        judge_model = args.judge_model or args.model
    if not args.data_dir:
        args.data_dir = (
            old_meta.get("data_dir") if old_meta else None
        ) or DEFAULT_DATA_DIR
    data = load_data(args.data_dir)

    # One client serves both the candidate model and the judge.
    client = OpenAI(
        api_key=api_key, base_url=args.base_url,
        timeout=180.0, max_retries=2,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=False)

    metadata = {
        **(old_meta or {}),
        **vars(args),
        "judge_model_effective": judge_model,
        "source": data["source"],
        "prompts": data["prompts"],
        "judge_instructions": data["judge_instructions"],
        "cases": data["cases"],
    }
    if old_meta:
        # Keep the generating model, settings, and prompts of the old run.
        for key in ("model", "temperature", "max_tokens", "trials",
                    "order_seed", "prompts"):
            metadata[key] = old_meta[key]
        # Runs from before this option sent no reasoning setting.
        metadata["reasoning"] = old_meta.get("reasoning", "default")
        metadata["regraded_from"] = args.regrade
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.regrade:
        records = regrade(
            client, args, parser, judge_model, data, out, old_meta
        )
    else:
        records = generate(client, args, judge_model, data, out)

    summary = analysis.summarize(records)
    analysis.write_summary(out, summary)

    print("\nSUMMARY (graded rows only; see the analysis below for fair "
          "comparisons)")
    for row in summary:
        rate = (
            f"{row['pass_rate']:.1%}"
            if row["pass_rate"] is not None else "N/A"
        )
        print(
            f"{row['variant']:20} {row['category']:20} "
            f"pass={row['passed']}/{row['graded']} ({rate}), "
            f"unsupported={row['unsupported_claims']}, "
            f"unnecessary_abstentions={row['unnecessary_abstentions']}, "
            f"errors={row['errors']} (empty={row['empty_answers']})"
        )

    report = analysis.build_report(records, metadata)
    (out / "analysis.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print()
    analysis.print_report(report)
    print(f"\nResults saved in: {out}")


if __name__ == "__main__":
    main()
