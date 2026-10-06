"""Tri-state extraction for MIMIC cases: the Paper 1 prompt, schema and parser, run on the
selected discharge-note sections by a LOCAL model only, cache-first.

`structured_oracle` needs no model: it returns the structured truth as if the note documented
every recorded value (False booleans as Absent). It is an upper bound for dry runs and tests,
not an estimate of real notes.
"""

from collections.abc import Mapping
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
from calc_bounds.types import BoolDomain, DocumentedState

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
