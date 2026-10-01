import matplotlib.pyplot as plt
import numpy as np

steps = [2000,4000,6000,8000,10000,12000,14000,
         16000,18000,20000,24000,28000,30000]

val_loss = [3.9103,3.6211,3.4841,3.4048,3.3534,
            3.3316,3.2851,3.2827,3.2625,
            3.2368,3.2191,3.1286,3.1038]

plt.figure(figsize=(7,4.5))

plt.plot(steps, val_loss, marker='o', linewidth=2, color='#1f4e79',
         label='Final 98.9M model')

plt.axhline(4.06, linestyle='--', color='#9e9e9e', label='6L/384 nanoGPT')
plt.axhline(3.82, linestyle='--', color='#f28e2b', label='8L/512 nanoGPT')
plt.axhline(3.71, linestyle='--', color='#59a14f', label='8L/512 Muon')
plt.axhline(3.38, linestyle='--', color='#b07aa1', label='10L/640 Muon')

plt.xlabel("Training Step")
plt.ylabel("Validation Loss")
plt.title("Optimization and Convergence Progression")
plt.legend(title="Final val loss of earlier stages", fontsize=8, title_fontsize=8)
plt.tight_layout()
plt.savefig("training_curves.pdf")
plt.savefig("figures/training_curves.png", dpi=150)
