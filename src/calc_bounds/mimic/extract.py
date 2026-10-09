"""Tri-state extraction for MIMIC cases: the Paper 1 prompt, schema and parser, run on the
selected discharge-note sections by a LOCAL model only, cache-first.

`structured_oracle` needs no model: it returns the structured truth as if the note documented
every recorded value (False booleans as Absent). It is an upper bound for dry runs and tests,
not an estimate of real notes.
"""

import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from calc_bounds.calculators import Calculator
from calc_bounds.config import ProviderConfig
from calc_bounds.extraction import ExtractionResult
from calc_bounds.extraction.llm import (
    EXTRACT_SYSTEM,
    EXTRACTION_PROMPT_VERSION,
    extraction_prompt,
    extraction_schema,
    parse_extraction,
)
from calc_bounds.extraction.oracle import oracle_extractions
from calc_bounds.llm import LLM, DiskCache, LLMClient, LLMRequest
from calc_bounds.llm.clients import make_client
from calc_bounds.mimic.config import assert_local_provider
from calc_bounds.mimic.notes import SectionedNote
from calc_bounds.mimic.truth import MimicCase
from calc_bounds.types import (
    BoolDomain,
    DocumentedState,
    EvidenceSpan,
    NumericDomain,
    OrdinalDomain,
    Present,
)

NOTE_SOURCE = "mimic-discharge-sections"


def structured_oracle(calc: Calculator, case: MimicCase) -> ExtractionResult:
    documented = {
        p: DocumentedState.NEGATIVE
        if isinstance(calc.param(p).domain, BoolDomain) and v is False
        else DocumentedState.POSITIVE
        for p, v in case.truth.items()
    }
    values = oracle_extractions(calc.parameters, case.truth, documented)
    return ExtractionResult(case_id=case.case_id, values=values, extractor="oracle")


def local_llm(providers: Mapping[str, ProviderConfig], cache_dir: Path, ledger: Path) -> LLM:
    """An `LLM` whose clients are created only for local providers (checked again here)."""

    def client(name: str) -> LLMClient:
        assert_local_provider(name, providers[name])
        return make_client(providers[name])

    return LLM(client, DiskCache(cache_dir), max_cost_usd=0.0, ledger_path=ledger)


def extraction_request(pcfg: ProviderConfig, calc: Calculator, note: str) -> LLMRequest:
    params = pcfg.request_params() | {"format": extraction_schema(calc)}
    if pcfg.kind == "ollama":
        params |= {"logprobs": True, "top_logprobs": 3}
    return LLMRequest(
        provider=pcfg.kind,
        model=pcfg.model,
        system=EXTRACT_SYSTEM,
        messages=[{"role": "user", "content": extraction_prompt(note, calc)}],
        params=params,
    )


def llm_extract(
    cases: list[MimicCase],
    notes: Mapping[int, SectionedNote],
    calcs: Mapping[str, Calculator],
    *,
    extractor: str,
    provider: str,
    pcfg: ProviderConfig,
    llm: LLM,
    max_workers: int,
) -> list[ExtractionResult]:
    assert_local_provider(provider, pcfg)

    def work(case: MimicCase) -> ExtractionResult:
        calc = calcs[case.calculator]
        note = notes[case.hadm_id].text
        req = extraction_request(pcfg, calc, note)
        resp = llm.complete(req, provider=provider, stage=f"mimic-extract:{extractor}")
        result = parse_extraction(case.case_id, note, calc, resp.text, resp.logprobs)
        return result.model_copy(
            update={
                "usage": resp.usage,
                "extractor": f"{extractor}@{EXTRACTION_PROMPT_VERSION}",
                "render": NOTE_SOURCE,
            }
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(work, cases))


# --- Post-extraction rules (code, not model) ---------------------------------------------------

STRUCTURED_HEADER = "\n\n[Structured record]"
_BUN = re.compile(r"\bBUN\b|\burea\s*n(?:itrogen)?\b", re.IGNORECASE)
"""BUN as written in US notes: 'BUN 28', and MIMIC lab lines 'UreaN-28' / 'urea nitrogen'."""
_ANALYTE_UNITS = {"urea_mg/dL", "bun_mg/dL"}


def bun_unit_rule(ext: ExtractionResult) -> ExtractionResult:
    """Urea claims whose evidence quotes BUN ('BUN', 'UreaN', 'urea nitrogen') are BUN in
    mg/dL (US reporting), whatever unit the model returned. MIMIC lab lines write 'UreaN-28'
    with no unit; reading it as urea mmol/L inflated urea about 2.8-fold in the pilot.
    Decided by code from the evidence span."""
    e = ext.values.get("urea")
    if not isinstance(e, Present) or e.unit in _ANALYTE_UNITS or not _BUN.search(e.evidence.text):
        return ext
    return ext.model_copy(
        update={"values": ext.values | {"urea": e.model_copy(update={"unit": "bun_mg/dL"})}}
    )


def with_structured_params(
    calc: Calculator, case: MimicCase, ext: ExtractionResult, note_text: str, params: Sequence[str]
) -> tuple[ExtractionResult, str]:
    """Give `params` (age, sex) from the structured record, as any EHR shows them beside the
    note. MIMIC-IV-Note masks ages, so the note alone can never state them. The values are
    appended to the note text as a short '[Structured record]' line so every claim keeps an
    evidence span that is an exact substring of the text the policies see."""
    ids = {p.id for p in calc.parameters}
    text, values = note_text, dict(ext.values)
    for pid in params:
        if pid not in ids or pid not in case.truth:
            continue
        spec, v = calc.param(pid), case.truth[pid]
        match spec.domain:
            case OrdinalDomain(levels=levels):
                piece, unit = f"{spec.label}: {levels[int(v)]}", None
            case NumericDomain(unit=u):
                piece, unit = f"{spec.label}: {float(v):g} {u}", u
            case BoolDomain():
                piece, unit = f"{spec.label}: {'yes' if v else 'no'}", None
        if text == note_text:
            text += STRUCTURED_HEADER
        start = len(text) + 1
        text += " " + piece + "."
        values[pid] = Present(
            value=v,
            unit=unit,
            confidence=1.0,
            confidence_source="oracle",
            evidence=EvidenceSpan(start=start, end=start + len(piece), text=piece),
        )
    return ext.model_copy(update={"values": values}), text
