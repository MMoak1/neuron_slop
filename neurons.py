"""
Neuromorphic scratchpad — just messing around with spiking neurons.

Leaky Integrate-and-Fire (LIF) neurons wired together with simple synapses.
Nothing fancy, just seeing spikes happen.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Simulation params ──────────────────────────────────────────
dt = 0.1       # time step (ms)
T = 500.0      # total simulation time (ms)
steps = int(T / dt)
time = np.arange(0, T, dt)

# ── LIF Neuron parameters ─────────────────────────────────────
V_REST = -70.0    # mV, resting potential
V_THRESH = -55.0  # mV, spike threshold
V_RESET = -75.0   # mV, post-spike reset (slight undershoot)
TAU_M = 15.0      # ms, membrane time constant
R_M = 10.0        # MΩ, membrane resistance
T_REF = 3.0       # ms, refractory period


class Neuron:
    """A single leaky integrate-and-fire neuron."""

    def __init__(self, name=""):
        self.name = name
        self.v = V_REST           # membrane potential
        self.ref_timer = 0.0      # time left in refractory period
        self.spiked = False       # did it spike this step?

        # recording
        self.v_history = []
        self.spike_times = []

    def step(self, I_input, t):
        """Advance one time step. I_input is total current in nA."""
        self.spiked = False

        if self.ref_timer > 0:
            # still in refractory period — clamp to reset
            self.ref_timer -= dt
            self.v = V_RESET
        else:
            # LIF dynamics: τ dV/dt = -(V - V_rest) + R*I
            dv = (-(self.v - V_REST) + R_M * I_input) / TAU_M
            self.v += dv * dt

            # spike?
            if self.v >= V_THRESH:
                self.spiked = True
                self.spike_times.append(t)
                self.v = V_RESET
                self.ref_timer = T_REF

        self.v_history.append(self.v)
        return self.spiked


class Synapse:
    """A simple synapse connecting two neurons."""

    def __init__(self, pre: Neuron, post: Neuron, weight=0.5, delay_ms=1.0):
        self.pre = pre
        self.post = post
        self.weight = weight  # synaptic weight (nA of current delivered)
        self.delay = int(delay_ms / dt)  # delay in time steps
        self.spike_buffer = []  # buffer of pending spike deliveries

    def step(self):
        """Check for pre spikes and deliver current after delay."""
        if self.pre.spiked:
            self.spike_buffer.append(self.delay)

        current = 0.0
        new_buffer = []
        for remaining in self.spike_buffer:
            if remaining <= 0:
                current += self.weight
            else:
                new_buffer.append(remaining - 1)
        self.spike_buffer = new_buffer
        return current  # current to add to post neuron


# ── Build a small network ──────────────────────────────────────
# 
#   [Input A] ──→ [Inter 1] ──→ [Output]
#   [Input B] ──→ [Inter 2] ──↗
#
# Input neurons get external drive, interneurons relay to output.

input_a = Neuron("Input A")
input_b = Neuron("Input B")
inter_1 = Neuron("Inter 1")
inter_2 = Neuron("Inter 2")
output  = Neuron("Output")

neurons = [input_a, input_b, inter_1, inter_2, output]

synapses = [
    Synapse(input_a, inter_1, weight=1.8, delay_ms=1.5),
    Synapse(input_b, inter_2, weight=1.8, delay_ms=1.5),
    Synapse(inter_1, output,  weight=1.2, delay_ms=2.0),
    Synapse(inter_2, output,  weight=1.2, delay_ms=2.0),
    # lateral inhibition between interneurons (negative weight)
    Synapse(inter_1, inter_2, weight=-0.8, delay_ms=1.0),
    Synapse(inter_2, inter_1, weight=-0.8, delay_ms=1.0),
]

# ── External drive ─────────────────────────────────────────────
# Input A: steady current with some noise
# Input B: current pulse from t=150 to t=350

def external_current(neuron, t):
    """External input current for a given neuron at time t (ms)."""
    if neuron is input_a:
        # constant drive + noise
        return 1.8 + np.random.normal(0, 0.3)
    elif neuron is input_b:
        # pulse of current
        if 150 <= t <= 350:
            return 2.2 + np.random.normal(0, 0.2)
        else:
            return 0.3 + np.random.normal(0, 0.1)
    else:
        return 0.0  # only driven by synapses


# ── Run simulation ─────────────────────────────────────────────
print(f"Simulating {len(neurons)} neurons for {T}ms...")

for step_i in range(steps):
    t = step_i * dt

    # compute synaptic currents first
    syn_currents = {n: 0.0 for n in neurons}
    for syn in synapses:
        current = syn.step()
        syn_currents[syn.post] += current

    # step each neuron
    for n in neurons:
        I = external_current(n, t) + syn_currents[n]
        n.step(I, t)

print("Done!\n")

# ── Print spike counts ─────────────────────────────────────────
for n in neurons:
    rate = len(n.spike_times) / (T / 1000.0)  # Hz
    print(f"  {n.name:10s}: {len(n.spike_times):4d} spikes ({rate:.1f} Hz)")

# ── Plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(neurons) + 1, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Spiking Neural Network — LIF Neurons", fontsize=14, fontweight='bold')

colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db', '#9b59b6']

# membrane potential traces
for i, n in enumerate(neurons):
    ax = axes[i]
    ax.plot(time, n.v_history, color=colors[i], linewidth=0.6, alpha=0.9)
    ax.axhline(V_THRESH, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_ylabel(f"{n.name}\n(mV)", fontsize=8)
    ax.set_ylim(V_RESET - 5, V_THRESH + 10)
    ax.tick_params(labelsize=7)

    # mark spikes
    for st in n.spike_times:
        ax.axvline(st, color=colors[i], alpha=0.3, linewidth=0.5)

# raster plot at bottom
ax_raster = axes[-1]
for i, n in enumerate(neurons):
    if n.spike_times:
        ax_raster.scatter(n.spike_times, [i] * len(n.spike_times),
                          s=2, color=colors[i], marker='|', linewidths=1.5)
ax_raster.set_yticks(range(len(neurons)))
ax_raster.set_yticklabels([n.name for n in neurons], fontsize=8)
ax_raster.set_xlabel("Time (ms)")
ax_raster.set_ylabel("Neuron")
ax_raster.set_title("Spike Raster", fontsize=10)

plt.tight_layout()
plt.savefig("/home/max/CODE/projects/gen_ai/neurons_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to neurons_plot.png")
