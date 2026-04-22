"""A/B experiment module — comparative validation of the MDK Fleet agent layer.

Run with:
    python -m ab_experiment.runner --scenario smoke --duration-min 1
    python -m ab_experiment.runner --scenario full --duration-min 60

Track A: full agent fleet (simulator -> ingest -> det_tools -> Maestro -> action)
Track B: deterministic-only baseline (same sim seed, rule-based action mapping)
"""
