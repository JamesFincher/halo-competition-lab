# Staged Exp162: Qwen3 sparse-edit width dose response

Intentionally inactive (`staged_*.py`) until exp152's H22 arms are interpreted. Exp152 uses
64 down-projection columns as its first 4B H22-style operating point, whereas the validated
135M ultra-sparse sweet spot was 16 columns. This no-training ablation reloads Qwen3-4B and
measures edit-only behavioral/continuation effects at 16/32/64/128 columns so an H22 failure
or success at 64 can be interpreted as a width/dose effect rather than a composition verdict.
