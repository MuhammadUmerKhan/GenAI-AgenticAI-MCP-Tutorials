# Guardrails AI — A Hands-On Tutorial

> A single-file guide to [Guardrails AI](https://guardrailsai.com/guardrails/docs/concepts/hub):
> validators that check an LLM's **input and output**, composed into a **Guard** that runs
> inside your app's critical path. Read top to bottom — each section explains *what a piece does*
> and *why*, then shows the code.

---

## Table of contents

1. [Core concepts](#1-core-concepts)
2. [One-time setup](#2-one-time-setup)
3. [Step 1 — Your first Guard (no LLM)](#step-1--your-first-guard-no-llm)
4. [Step 2 — OnFailAction: what happens on failure](#step-2--onfailaction-what-happens-on-failure)
5. [Step 3 — Stacking multiple validators](#step-3--stacking-multiple-validators)
6. [Step 4 — Guarding a real LLM call](#step-4--guarding-a-real-llm-call)
7. [Step 5 — Validated structured output (Pydantic)](#step-5--validated-structured-output-pydantic)
8. [Step 6 — Practical production validators](#step-6--practical-production-validators)
9. [Cheat sheet](#9-cheat-sheet)

---

## 1. Core concepts

Guardrails AI has four ideas. Learn these four and everything else is composition.

| Term | What it is | Analogy |
|------|-----------|---------|
| **Validator** | A single check on text: regex, length, PII, toxicity, ... | One unit test |
| **Guard** | One or more validators wrapped together; you run text (or a whole LLM call) through it | A test suite |
| **Hub** | The registry where validators live. You install them by URI: `guardrails hub install hub://guardrails/<name>` | pip for validators |
| **OnFailAction** | What a validator does when it fails: `noop` / `exception` / `fix` / `filter` / `reask` | A policy |

**The mental model:** a Guard sits between your app and the model. Text flows in → validators
run → based on `OnFailAction`, the Guard passes it, fixes it, blocks it, or asks the model to
retry. Nothing reaches your user until the Guard says it's okay.

```
user ──▶ [ Guard: validator₁ · validator₂ · ... ] ──▶ LLM ──▶ [ same Guard on output ] ──▶ user
```

---

## 2. One-time setup

### 2a. Get a Hub token (free)

Validators are distributed through the Hub, so you authenticate once.

1. Create a free token: <https://hub.guardrailsai.com/tokens>
2. Configure the CLI (writes `~/.guardrailsrc`):

   ```bash
   guardrails configure
   ```

   It prompts for metrics reporting, remote-inference preference, and your token. Defaults are fine.

### 2b. LLM key

This tutorial calls **Groq** through LiteLLM (which Guardrails uses under the hood). Your `.env`
already has `GROQ_API_KEY`. Any provider works via the LiteLLM `"provider/model"` string, e.g.
`"groq/llama-3.3-70b-versatile"`.

### 2c. Installing validators

Each step tells you exactly which validator to install. Example:

```bash
guardrails hub install hub://guardrails/regex_match --quiet
```

- Rule-based validators (regex, length, competitor check) install instantly.
- ML-based validators (`toxic_language`, `detect_pii`) download a small local model on first
  install, so those take ~a minute and pull extra dependencies.

After installing, the validator becomes importable from `guardrails.hub`.

> **Windows note:** the installer prints a ✅ emoji that the default console encoding (`charmap`)
> can't render, which crashes the CLI with `'charmap' codec can't encode character '✅'`.
> The fix is to force UTF-8 for the command:
> ```bash
> PYTHONUTF8=1 guardrails hub install hub://guardrails/regex_match
> ```
> (In PowerShell: `$env:PYTHONUTF8=1; guardrails hub install ...`)

---

## Step 1 — Your first Guard (no LLM)

**Purpose:** understand a Guard in isolation, with zero LLM involvement. We validate plain strings
against a regex: "a single capitalized word."

**Install:**
```bash
guardrails hub install hub://guardrails/regex_match --quiet
```

```python
from guardrails import Guard, OnFailAction
from guardrails.hub import RegexMatch  # importable AFTER `guardrails hub install`

# Guard()          -> creates an empty Guard.
# .use(validator)  -> attaches a validator to it. Returns the Guard, so calls chain.
# RegexMatch(...)  -> passes only if the text fully matches the regex.
# on_fail=NOOP     -> IMPORTANT: the DEFAULT on_fail for most validators is `exception`,
#                     which RAISES on failure. We want .parse() to return a False verdict
#                     instead of throwing, so we set NOOP explicitly. (More on this in Step 2.)
guard = Guard().use(
    RegexMatch(regex="^[A-Z][a-z]*$", on_fail=OnFailAction.NOOP)
)

def check(text: str) -> None:
    # .parse(text) runs the validators against the text. NO LLM call happens here —
    # it just treats `text` as if it were model output and validates it.
    result = guard.parse(text)
    # .validation_passed -> True only if EVERY validator passed.
    status = "PASS ✅" if result.validation_passed else "FAIL ❌"
    print(f"{status}  {text!r}")

check("Caesar")        # matches           -> PASS
check("Caesar Salad")  # contains a space  -> FAIL
check("caesar")        # lowercase start   -> FAIL
```

**Expected output:**
```
PASS ✅  'Caesar'
FAIL ❌  'Caesar Salad'
FAIL ❌  'caesar'
```

**What to take away:**
- `Guard()` → `.use()` → `.parse()` is the whole lifecycle.
- `.parse()` = "validate this text." `.validation_passed` = the boolean verdict.

---

## Step 2 — OnFailAction: what happens on failure

**Purpose:** a validator failing is not the end — *you decide the consequence* with `on_fail=`.
This is the single most important knob in Guardrails.

| Action | Behavior | When to use |
|--------|----------|-------------|
| `NOOP` | Record the failure, return text unchanged | Logging / observability |
| `EXCEPTION` | Raise an error, stop everything | Production fail-closed: bad output must be blocked |
| `FIX` | Auto-correct to a passing value (validator-specific) | You'd rather sanitize than reject |
| `FILTER` | Drop the failing value | Structured output where one bad field shouldn't kill the rest |
| `REASK` | Ask the LLM to try again | Only meaningful when a model is in the loop (Step 4) |

**Install:**
```bash
guardrails hub install hub://guardrails/valid_length --quiet
```

```python
from guardrails import Guard, OnFailAction
from guardrails.hub import ValidLength  # passes if min <= len(text) <= max

TEXT = "this string is definitely far too long for the limit"  # 52 chars, limit is 10

# 1) NOOP — never raises. You inspect the verdict yourself.
noop_guard = Guard().use(ValidLength(min=1, max=10, on_fail=OnFailAction.NOOP))
res = noop_guard.parse(TEXT)
print("NOOP      -> validation_passed:", res.validation_passed)   # False

# 2) EXCEPTION — fail-closed. The go-to for production safety checks.
strict_guard = Guard().use(ValidLength(min=1, max=10, on_fail=OnFailAction.EXCEPTION))
try:
    strict_guard.parse(TEXT)
except Exception as e:
    print("EXCEPTION -> raised:", type(e).__name__)               # ValidationError

# 3) FIX — auto-corrects. ValidLength's fix truncates to `max`.
fix_guard = Guard().use(ValidLength(min=1, max=10, on_fail=OnFailAction.FIX))
fixed = fix_guard.parse(TEXT)
# .validated_output -> the (possibly corrected) text the Guard is willing to return.
print("FIX       -> corrected output:", repr(fixed.validated_output))
```

**Expected output:**
```
NOOP      -> validation_passed: False
EXCEPTION -> raised: ValidationError
FIX       -> corrected output: 'this strin'
```

**What to take away:**
- `.validated_output` is the text the Guard *returns* (after any `FIX`); `.validation_passed` is
  whether it was clean to begin with.
- Same validator, three very different behaviors — chosen entirely by `on_fail`.

---

## Step 3 — Stacking multiple validators

**Purpose:** real guards combine checks. A Guard with several validators passes only if **all** of
them pass.

> ⚠️ **Gotcha:** pass every validator to a **single** `.use(...)` call. Calling `.use()` twice —
> `Guard().use(A).use(B)` — does **not** stack them; the second call **replaces** the first, so
> `A` is silently dropped and never runs. Always write `Guard().use(A, B)`.

**Install:**
```bash
guardrails hub install hub://guardrails/regex_match --quiet
guardrails hub install hub://guardrails/valid_length --quiet
```

```python
from guardrails import Guard, OnFailAction
from guardrails.hub import RegexMatch, ValidLength

# Rule: a single capitalized word (regex) AND 1..12 characters (length).
# BOTH validators go in ONE .use(...) call so both stay active (see gotcha above).
guard = Guard().use(
    RegexMatch(regex="^[A-Z][a-z]*$", on_fail=OnFailAction.NOOP),
    ValidLength(min=1, max=12, on_fail=OnFailAction.NOOP),
)

def check(text: str) -> None:
    res = guard.parse(text)
    print(f"{'PASS ✅' if res.validation_passed else 'FAIL ❌'}  {text!r}")

check("Caesar")               # word ✅ + length ✅        -> PASS
check("Supercalifragilistic") # word ✅ but 20 chars      -> FAIL (length)
check("caesar")               # length ✅ but lowercase   -> FAIL (regex)
```

**Expected output:**
```
PASS ✅  'Caesar'
FAIL ❌  'Supercalifragilistic'
FAIL ❌  'caesar'
```

**What to take away:** validators are **AND-composed**. Order doesn't change the verdict, but it
can change which failure you see first when using `EXCEPTION`.

---

## Step 4 — Guarding a real LLM call

**Purpose:** this is the payoff. Instead of `guard.parse(text)`, you **call the Guard like a
function**. The Guard makes the LLM call for you, then validates the model's output. With
`on_fail=REASK`, it automatically re-prompts the model when a check fails.

**Install:**
```bash
guardrails hub install hub://guardrails/toxic_language --quiet
```

```python
import os
from dotenv import load_dotenv
from guardrails import Guard, OnFailAction
from guardrails.hub import ToxicLanguage

load_dotenv()
assert os.getenv("GROQ_API_KEY"), "Set GROQ_API_KEY in your .env"

# ToxicLanguage runs a small LOCAL classifier over the output.
#   threshold=0.5           -> how confident before flagging as toxic (0..1)
#   validation_method="sentence" -> score each sentence (vs. the whole block)
#   on_fail=REASK           -> if toxic, Guardrails re-prompts the model automatically
guard = Guard().use(
    ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail=OnFailAction.REASK)
)

# Calling the guard(...) runs the LLM call THROUGH the validators.
# `model` uses LiteLLM's "provider/model" format; `messages` is the usual chat list.
result = guard(
    model="groq/llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a polite, helpful assistant."},
        {"role": "user",   "content": "Write one friendly sentence welcoming a new teammate."},
    ],
)

print("Validation passed:", result.validation_passed)
print("Guarded output:\n", result.validated_output)
```

**Expected output** (wording varies — the model generates it):
```
Validation passed: True
Guarded output:
 Welcome to the team — we're so glad to have you here and can't wait to work with you!
```

**What to take away:**
- `guard.parse(text)` = validate text you already have.
  `guard(model=..., messages=...)` = let the Guard *make* the call and validate the result.
- `REASK` closes the loop: the model gets another chance instead of you just rejecting it.

---

## Step 5 — Validated structured output (Pydantic)

**Purpose:** force the model to return **JSON that matches a schema**, and validate individual
fields. `Guard.for_pydantic(Model)` builds the Guard from your Pydantic model; validators attached
to a field via `validators=` run on that field's value.

**Install:**
```bash
guardrails hub install hub://guardrails/valid_length --quiet
```

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from guardrails import Guard, OnFailAction
from guardrails.hub import ValidLength

load_dotenv()
assert os.getenv("GROQ_API_KEY"), "Set GROQ_API_KEY in your .env"

class Pet(BaseModel):
    pet_type: str = Field(description="Species of the pet, e.g. dog or cat")
    # `validators=` attaches a Guardrails validator to THIS field only.
    # If the name is too long, REASK makes the model regenerate.
    name: str = Field(
        description="A short, unique pet name",
        validators=[ValidLength(min=1, max=12, on_fail=OnFailAction.REASK)],
    )

# .for_pydantic() -> builds a Guard whose job is "return JSON shaped like Pet, validated".
guard = Guard.for_pydantic(output_class=Pet)

result = guard(
    model="groq/llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Invent a pet. Return its type and a short name."},
    ],
)

print("Validation passed:", result.validation_passed)
print("Parsed object    :", result.validated_output)  # dict matching the Pet schema
```

**Expected output** (values vary):
```
Validation passed: True
Parsed object    : {'pet_type': 'dog', 'name': 'Biscuit'}
```

**What to take away:** the `description=` on each field is fed to the model as instructions, and
the field's `validators=` enforce the rules. You get typed, checked data instead of a raw string.

---

## Step 6 — Practical production validators

**Purpose:** three guardrails you'll genuinely reach for. Each block is independent — comment any
out. This uses `.parse()` on sample strings so you can see the effect without spending LLM calls,
but each of these works identically as an output guard on a real `guard(...)` call.

**Install:**
```bash
guardrails hub install hub://guardrails/detect_pii --quiet
guardrails hub install hub://guardrails/toxic_language --quiet
guardrails hub install hub://guardrails/competitor_check --quiet
```

```python
from guardrails import Guard, OnFailAction
from guardrails.hub import DetectPII, ToxicLanguage, CompetitorCheck

def show(label: str, guard: Guard, text: str) -> None:
    res = guard.parse(text)
    print(f"\n[{label}]  input: {text!r}")
    print("  passed:", res.validation_passed)
    print("  output:", repr(res.validated_output))

# 1) PII — DetectPII flags personal data. FIX anonymizes it instead of raising.
#    pii_entities uses Microsoft Presidio entity names (EMAIL_ADDRESS, PHONE_NUMBER, PERSON, ...).
pii_guard = Guard().use(
    DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail=OnFailAction.FIX)
)
show("PII", pii_guard, "Reach me at jane.doe@example.com or 415-555-0199.")

# 2) Toxicity — NOOP so we just see the flag without raising.
tox_guard = Guard().use(
    ToxicLanguage(threshold=0.5, validation_method="sentence", on_fail=OnFailAction.NOOP)
)
show("Toxicity", tox_guard, "You are amazing, thanks for the help!")

# 3) Competitor mentions — FIX strips sentences that name a competitor.
comp_guard = Guard().use(
    CompetitorCheck(competitors=["OpenAI", "Google"], on_fail=OnFailAction.FIX)
)
show("Competitor", comp_guard, "We use OpenAI models, but our tooling is our own.")
```

**Expected output** (anonymized/stripped forms vary slightly by version):
```
[PII]  input: 'Reach me at jane.doe@example.com or 415-555-0199.'
  passed: False
  output: 'Reach me at <EMAIL_ADDRESS> or <PHONE_NUMBER>.'

[Toxicity]  input: 'You are amazing, thanks for the help!'
  passed: True
  output: 'You are amazing, thanks for the help!'

[Competitor]  input: 'We use OpenAI models, but our tooling is our own.'
  passed: False
  output: 'but our tooling is our own.'
```

**What to take away:** the same three-line pattern — `Guard().use(SomeValidator(..., on_fail=...))`
— scales from a toy regex to production PII redaction. The validator changes; the shape doesn't.

---

## 9. Cheat sheet

**Lifecycle**
```python
guard = Guard().use(Validator(..., on_fail=OnFailAction.X))  # build
guard.parse("some text")                                     # validate text you have
guard(model="groq/...", messages=[...])                      # let the Guard call the LLM
```

**Result object**
| Attribute | Meaning |
|-----------|---------|
| `.validation_passed` | `True` if every validator passed |
| `.validated_output`  | The text/dict the Guard returns (after any `FIX`) |
| `.raw_llm_output`     | The model's original, pre-validation output |

**OnFailAction quick pick**
- Block bad output → `EXCEPTION`
- Sanitize and continue → `FIX`
- Give the model another try → `REASK`
- Just log → `NOOP`

**Installing a validator**
```bash
guardrails hub install hub://guardrails/<name> --quiet
# then: from guardrails.hub import <ClassName>
```

**Where to explore more validators:** <https://hub.guardrailsai.com>
