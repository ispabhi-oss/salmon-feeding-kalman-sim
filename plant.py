# plant.py  --  Stages 1 and 2 of the salmon feeding-loop simulation
# Stage 1: a hidden "appetite" that drifts, plus a noisy sensor that measures it.
# Stage 2: two feeding controllers (fixed and reactive) compared on feed eaten, wasted and missed.
# All numbers are ASSUMED values for learning, not real fish data.

import numpy as np                 # numerical arrays and random numbers
import matplotlib.pyplot as plt    # plotting

# ---------------------------------------------------------------
# SETTINGS (change these to experiment)
# ---------------------------------------------------------------
TYPICAL = 0.7      # the appetite the fish drift back towards (0 = not hungry, 1 = very hungry)
THETA = 0.02       # how fast appetite returns to TYPICAL each minute (mean reversion)
SIGMA_A = 0.015    # size of the random push on appetite each minute (real, hidden change)
SIGMA_Z = 0.2      # size of the sensor noise (measurement error, not a real change)
N_MINUTES = 480    # simulate 8 hours, one step per minute
A0 = 0.7           # appetite at minute 0
C_MAX = 10.0       # most feed the fish can eat per minute when appetite = 1 (arbitrary units)

rng = np.random.default_rng(42)    # fixed seed so every run gives the same numbers

# ---------------------------------------------------------------
# STAGE 1a: simulate the hidden appetite (the "truth" the controller cannot see directly)
# ---------------------------------------------------------------
appetite = []          # will hold the true appetite for every minute
a = A0                 # current appetite
for t in range(N_MINUTES):
    # pull back towards TYPICAL, then add a small random change
    a = a + THETA * (TYPICAL - a) + SIGMA_A * rng.normal()
    a = np.clip(a, 0.0, 1.0)       # appetite must stay between 0 and 1
    appetite.append(a)

# ---------------------------------------------------------------
# STAGE 1b: the noisy sensor = truth + random error
# ---------------------------------------------------------------
readings = []          # what the sensor reports each minute
for t in range(N_MINUTES):
    readings.append(appetite[t] + SIGMA_Z * rng.normal())

# ---------------------------------------------------------------
# STAGE 2a: the controllers. Each returns how much feed to give THIS minute.
# ---------------------------------------------------------------
def fixed_controller(t):
    # ignores the sensor: always feeds 70 % of the maximum
    return 0.7 * C_MAX

def reactive_controller(t):
    # uses the PREVIOUS minute's sensor reading (the current one is not known yet)
    if t == 0:
        return 0.7 * C_MAX                         # no reading yet, so use the fixed rate
    return readings[t - 1] * C_MAX                 # feed in proportion to the previous reading

# ---------------------------------------------------------------
# STAGE 2b: run one whole feeding session with a given controller
# ---------------------------------------------------------------
def run_feeding(controller):
    total_eaten = 0.0      # feed the fish actually ate
    total_waste = 0.0      # feed given but not eaten (lost into the water)
    total_missed = 0.0     # feed the fish could have eaten but was not given
    for t in range(N_MINUTES):
        feed = controller(t)                       # what the controller decides
        capacity = C_MAX * appetite[t]             # what the fish want this minute
        eaten = min(feed, capacity)                # cannot eat more than offered or wanted
        total_eaten += eaten
        total_waste += max(feed - capacity, 0.0)   # over-feeding
        total_missed += max(capacity - feed, 0.0)  # under-feeding
    return total_eaten, total_waste, total_missed


# ---------------------------------------------------------------
# STAGE 3a: the scalar Kalman filter (a better guess of the hidden appetite)
# Idea: each minute, (1) PREDICT using the fish model, (2) UPDATE using the new noisy reading.
# ---------------------------------------------------------------
Q = SIGMA_A ** 2       # variance of the real appetite changes (process noise)
R = 0.5 ** 2      # variance of the sensor noise (measurement noise)

def kalman_filter(z):
    # z = list of sensor readings. Returns the filtered appetite estimate for every minute.
    est = A0           # starting guess of appetite
    P = 0.1            # starting uncertainty (variance) of that guess; large = "not sure"
    estimates = []     # estimate after seeing each reading
    for t in range(len(z)):
        # PREDICT: where should appetite be now, based on the model alone?
        pred = est + THETA * (TYPICAL - est)          # same mean reversion as the real fish
        P_pred = (1 - THETA) ** 2 * P + Q             # uncertainty grows by the process noise
        # UPDATE: blend the prediction with the new reading
        K = P_pred / (P_pred + R)                     # Kalman gain: 0 = trust model, 1 = trust sensor
        est = pred + K * (z[t] - pred)                # move towards the reading by a fraction K
        P = (1 - K) * P_pred                          # uncertainty shrinks after using the reading
        estimates.append(est)
    return estimates

estimates = kalman_filter(readings)    # run the filter over the whole session

# error of the raw sensor versus error of the filter (root mean square, lower = better)
rmse_raw = np.sqrt(np.mean((np.array(readings) - np.array(appetite)) ** 2))
rmse_kf = np.sqrt(np.mean((np.array(estimates) - np.array(appetite)) ** 2))
print(f"\nRMSE of raw sensor: {rmse_raw:.4f}   RMSE of Kalman estimate: {rmse_kf:.4f}")

# ---------------------------------------------------------------
# STAGE 3b: a controller that feeds based on the filtered estimate
# ---------------------------------------------------------------
def kalman_controller(t):
    if t == 0:
        return 0.7 * C_MAX                            # nothing known yet
    last = estimates[t - 1]                           # estimate using readings up to the previous minute
    predicted_now = last + THETA * (TYPICAL - last)   # predict this minute's appetite
    return float(np.clip(predicted_now, 0.0, 1.0)) * C_MAX

# ---------------------------------------------------------------
# STAGE 2c: compare the controllers and print a table
# ---------------------------------------------------------------
print(f"{'controller':<10} {'eaten':>8} {'waste':>8} {'missed':>8}")
for name, ctrl in [("fixed", fixed_controller), ("reactive", reactive_controller), ("kalman", kalman_controller)]:
    eaten, waste, missed = run_feeding(ctrl)
    print(f"{name:<10} {eaten:8.0f} {waste:8.0f} {missed:8.0f}")

# ---------------------------------------------------------------
# PLOT (kept at the END so the table prints before the window opens)
# ---------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(readings, label="sensor reading (noisy)", alpha=0.5)
plt.plot(appetite, label="true appetite (hidden)", linewidth=2)
plt.plot(estimates, label="Kalman estimate", linewidth=2)
plt.xlabel("minute")
plt.ylabel("appetite (0 to 1)")
plt.legend()
plt.show()      # note the brackets: without them nothing opens