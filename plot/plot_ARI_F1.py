import matplotlib.pyplot as plt

# Dati
results = {
    '100': {'lambda': 0.5, 'ARI': 1.0, 'F1': 1.0},
    '1K': {'lambda': 0.99, 'ARI': 0.49, 'F1': 0.85},
    '10K': {'lambda': 0.992, 'ARI': 0.92, 'F1': 0.98}
}

# Ordina i dati in base alla dimensione
dimensions = ['100', '1K', '10K']
ARI_values = [results[d]['ARI'] for d in dimensions]
F1_values = [results[d]['F1'] for d in dimensions]
lambdas = [results[d]['lambda'] for d in dimensions]

# Crea la figura con due assi y
fig, ax1 = plt.subplots(figsize=(6, 6))
ax2 = ax1.twinx()

# Asse x: le dimensioni
x_positions = range(len(dimensions))

# ARI — linea arancione tratteggiata con marker 'o'
ax1.plot(
    x_positions,
    ARI_values,
    linestyle='dashed',
    marker='o',
    color='orange',
    label='ARI'
)

# F1 — linea verde solida con marker '^'
ax2.plot(
    x_positions,
    F1_values,
    linestyle='solid',
    marker='^',
    color='green',
    label='F1'
)

# Etichette degli assi
ax1.set_xlabel("Dataset Size")
ax1.set_ylabel("ARI", color='orange')
ax2.set_ylabel("F1 Score", color='green')

# Set tick labels
ax1.set_xticks(x_positions)
ax1.set_xticklabels(dimensions)

# Legenda con lambda per dimensione
legend_labels = [
    f"{dim} (λ={results[dim]['lambda']})" for dim in dimensions
]
legend_text = " | ".join(legend_labels)
plt.title(f"ARI and F1 vs Dataset Size\n{legend_text}")

# Aggiungi legenda per le linee
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(
    lines1 + lines2, 
    labels1 + labels2, 
    loc='lower center', 
    bbox_to_anchor=(0.5, -0.2), 
    ncol=2
)

fig.tight_layout()
plt.grid(True)
plt.show()
