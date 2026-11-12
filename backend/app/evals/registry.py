from app.evals.accuracy import AccuracyEval
from app.evals.base import Eval
from app.evals.latency import LatencyEval
from app.evals.reliability import ReliabilityEval
from app.evals.resources import ResourcesEval
from app.evals.tokenizer import TokenizerEval


def default_evals() -> dict[str, Eval]:
    """Dimension name -> evaluator. Adding a dimension means adding a class here."""
    evals = [AccuracyEval(), LatencyEval(), ResourcesEval(), TokenizerEval(), ReliabilityEval()]
    return {e.name: e for e in evals}
