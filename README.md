# Salmon feeding-loop simulation (self-study)

A Python simulation of a salmon pen. A hidden, mean-reverting appetite is
measured by a noisy sensor. Three feeding controllers (fixed, reactive and
Kalman-filter based) are compared on feed eaten, wasted and missed.

Run: python plant.py   (needs numpy and matplotlib)

Status: stages 1-3 done. Not yet included: sensor delay and a censored
pellet sensor.

This is an independent learning project using assumed parameters. It does not
use real fish data, and nothing in it comes from any employer, collaborator or
unpublished work.
