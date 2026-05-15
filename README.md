[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

# Lt-Adaptive-Collapse

**Adaptive Collapse Dynamics: A Unified Variational Principle for Stability, Degradation, and Collapse**

Implementation of the theoretical framework proposed by **Locus** (Stanislav Usychenko), 2026.

## About the Theory

This repository implements key concepts from the paper _Adaptive Collapse Dynamics_, where collapse, degradation, and stability are unified through a variational principle in an **extended state space** `(S(t), L(t))`.

**Core idea:** Systems minimize generalized free energy while consuming adaptive stability margin `L(t)`.

## Current Implementation

- `acd.py` — Adaptive Collapse Engine based on the `L²/t` scaling law (special regime of the theory)

## Quick Start

```python
from acd import AdaptiveCollapseEngine
import numpy as np

# Create engine
engine = AdaptiveCollapseEngine()

# Generate sample data
time_series = np.random.randn(1000)
adaptive_margin = np.ones(1000)  # L(t)

# Compute vitality scores
vitality = engine.compute_vitality(time_series, adaptive_margin)

# Detect collapse events
collapsed = engine.detect_collapse(vitality, threshold=0.1)