import MDAnalysis as mda
import numpy as np
import pandas as pd

u = mda.Universe("system.psf", "state_continue.dcd")
sel = u.select_atoms("not (resname HOH ION IAN)")

# ---- Calculate only last 100 frames ----
rg_last100 = []
t_last100 = []

for ts in u.trajectory[-1000:]:
    com = sel.center_of_mass()
    diff = sel.positions - com

    squared_distances = np.sum(diff**2, axis=1)
    weighted_sum_sq = np.sum(sel.masses * squared_distances)
    total_mass = np.sum(sel.masses)

    Rg = np.sqrt(weighted_sum_sq / total_mass)

    rg_last100.append(Rg)
    t_last100.append(ts.time)

rg_last100 = np.array(rg_last100)

# ---- Mean and std ----
rg_mean = np.mean(rg_last100)
rg_std = np.std(rg_last100, ddof=1)

print("Stats over last 100 frames:")
print(f"Average Rg = {rg_mean:.4f} Å")
print(f"Std dev Rg = {rg_std:.4f} Å")

# Save last-100 timeseries (optional)
df_last100 = pd.DataFrame({"Time_ps": t_last100, "Rg_Angstrom": rg_last100})
df_last100.to_csv("radius_of_gyration_last100.dat", sep=" ", index=False, header=False)

# ---- Distribution with fixed range 0–40 ----
bin_edges = np.arange(0, 40.0 + 0.5, 0.5)

counts, bin_edges = np.histogram(rg_last100, bins=bin_edges)

prob = counts / np.sum(counts)  # probability (sums to 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

dist_df = pd.DataFrame({
    "Rg_Angstrom": bin_centers,
    "Probability": prob
})
dist_df.to_csv("Rg_distribution.dat", sep=" ", index=False, header=False)

print("Saved distribution to: Rg_distribution.dat")
