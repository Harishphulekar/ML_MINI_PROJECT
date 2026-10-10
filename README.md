# PINN for Mechanics (UE24CS352A Mini-Project)
Physics-Informed Neural Network that (1) solves and (2) discovers parameters of a damped spring-mass system
`m x'' + c x' + k x = 0`, x(0)=1, x'(0)=0.

## Setup & run
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python pinn_mechanics.py     # trains both experiments, writes results.json + results.png
```
CPU is fine (a few minutes).

## What it does
- **Forward:** loss = ODE residual (autograd) + initial conditions. No data.
- **Inverse:** c, k are trainable (init 1.0, 1.0); trained on 40 noisy points + ODE residual. Compared to true c=0.4, k=4.0.
- Reference: closed-form underdamped solution.

## Files
`pinn_mechanics.py` code, `results.json` metrics, `results.png` plots, `writeup.pdf` report.
