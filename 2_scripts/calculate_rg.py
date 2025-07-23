import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

u = mda.Universe("polymer_drug_solvate.psf", "run_pr.dcd")

sel = u.select_atoms("not (name W HOH)")

rgyr_values = []
times = []

for ts in u.trajectory:
    com = sel.center_of_mass()
    
    positions = sel.positions
    masses = sel.masses
    
    diff = positions - com
    
    squared_distances = np.sum(diff**2, axis=1)
    
    weighted_sum_sq = np.sum(masses * squared_distances)
    
    total_mass = np.sum(masses)
    
    Rg = np.sqrt(weighted_sum_sq / total_mass)
    
    rgyr_values.append(Rg)
    times.append(ts.time)

last_20_avg = np.mean(rgyr_values[-20:])
print(f"Average Rg of last 20 frames: {last_20_avg:.2f} Å")
    
with open("radius_of_gyration_data.txt", "w") as f:
    f.write("Time (ps)\tRg (Å)\n")
    for t, rg in zip(times, rgyr_values):
        f.write(f"{t:.2f}\t{rg:.2f}\n")

plt.figure(figsize=(10, 6))
plt.plot(times, rgyr_values)
plt.xlabel("Time (ps)")
plt.ylabel("Radius of Gyration (Å)")
plt.title("Radius of Gyration over Time")
plt.grid(True)

plt.ylim(27, 37)
plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.savefig("radius_of_gyration.png")