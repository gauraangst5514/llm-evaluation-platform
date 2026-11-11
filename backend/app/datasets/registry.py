"""Static registry of evaluation datasets.

Each entry is a small, seeded, fixed subset of a public dataset. Licences below were
checked against the Hugging Face dataset card on 2026-11-12. Nothing is downloaded at
import time; loaders (M2+) fetch at runtime and never commit data to the repo.

Why only Global-MMLU for now: it is Apache-2.0, ungated, and the *same* questions
(shared `sample_id`) exist in English, Hindi and Bengali. That gives a genuinely
parallel subset for the quantization pilot, and the question text doubles as the
per-language tokenizer sample so fertility is measured on the exact text the model
is evaluated on. MILU (CC-BY-4.0) and FLORES-200 (CC-BY-SA-4.0) are gated on the
Hub; they can be added once accessed with a token and their fields verified.
"""

from dataclasses import asdict, dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Dataset

DEFAULT_SEED = 42
DEFAULT_N_ITEMS = 200

GLOBAL_MMLU_URL = "https://huggingface.co/datasets/CohereLabs/Global-MMLU"
GLOBAL_MMLU_FIELDS = {
    "id": "sample_id",
    "question": "question",
    "options": ["option_a", "option_b", "option_c", "option_d"],
    "answer": "answer",  # letter A-D
    "subject": "subject",
    "subject_category": "subject_category",
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_url: str
    license: str
    language: str
    split: str
    config_json: dict = field(default_factory=dict)
    n_items: int = DEFAULT_N_ITEMS
    seed: int = DEFAULT_SEED


def _global_mmlu(lang: str) -> DatasetSpec:
    return DatasetSpec(
        name=f"global_mmlu_{lang}",
        source_url=GLOBAL_MMLU_URL,
        license="Apache-2.0",
        language=lang,
        split="test",
        config_json={
            "kind": "mcq",
            "hf_repo": "CohereLabs/Global-MMLU",
            "hf_config": lang,
            "fields": GLOBAL_MMLU_FIELDS,
            # Entries sharing a parallel_group get the same sample_ids, chosen once
            # from the group's anchor language with the dataset seed.
            "parallel_group": "global_mmlu",
            "parallel_anchor": "en",
            "gated": False,
        },
    )


REGISTRY: tuple[DatasetSpec, ...] = (
    _global_mmlu("en"),
    _global_mmlu("hi"),
    _global_mmlu("bn"),
)


def get_spec(name: str) -> DatasetSpec | None:
    return next((s for s in REGISTRY if s.name == name), None)


async def sync_registry(session: AsyncSession) -> int:
    """Upsert registry entries by name. Returns how many rows were inserted.

    Existing rows keep their id and item_ids_json (so past runs stay reproducible);
    descriptive fields are refreshed from the registry.
    """
    existing = {d.name: d for d in (await session.scalars(select(Dataset))).all()}
    inserted = 0
    for spec in REGISTRY:
        values = asdict(spec)
        row = existing.get(spec.name)
        if row is None:
            session.add(Dataset(**values))
            inserted += 1
        else:
            for key, value in values.items():
                setattr(row, key, value)
    await session.commit()
    return inserted
