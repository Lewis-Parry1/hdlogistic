import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from to_be_titled.estimation import fit_diaconis_ylvisaker_logistic_regression
from to_be_titled.summary import summary

df = pd.read_csv("data/MultipleFeatures.csv")


if df['training'].dtype == object:
    train_mask = df['training'].astype(str).str.upper().isin(['TRUE', '1', 'T'])
else:
    train_mask = df['training'].astype(bool)


vars_all = [col for col in df.columns if 'fou' in col or 'kar' in col]


df.loc[train_mask, vars_all] = (
    df.loc[train_mask, vars_all] - df.loc[train_mask, vars_all].mean()
)


train_df = df[train_mask].copy()


y_train = (train_df['digit'].astype(str) == '7').astype(float).values.reshape(-1, 1)


X_full = train_df[vars_all].copy()
breakpoint()
X_full.insert(0, 'Intercept', 1.0)


nest_vars = [col for col in vars_all if 'fou' in col]
X_nest = train_df[nest_vars].copy()
X_nest.insert(0, 'Intercept', 1.0)


full_m = fit_diaconis_ylvisaker_logistic_regression(
    x=X_full.values,
    y=y_train,
    intercept_index=0
)
nest_m = fit_diaconis_ylvisaker_logistic_regression(
    x=X_nest.values,
    y=y_train,
    alpha=full_m.alpha,
    intercept_index=0
)


rescaled_coefs_full = summary(full_m, high_dimensional_correction=True)


orig_coefs = full_m.betas[1:].flatten()
resc_coefs = rescaled_coefs_full[1:].flatten()

valid_idx = np.where(orig_coefs != 0)[0][0]
mu_hat = orig_coefs[valid_idx] / resc_coefs[valid_idx]

is_kar = np.array(['kar' in col for col in vars_all])
colors = np.where(is_kar, 'teal', 'purple')

fig, ax = plt.subplots(figsize=(7, 7))

ax.scatter(
    orig_coefs, 
    resc_coefs, 
    facecolors=colors, 
    edgecolors=colors,
    alpha=0.25,
    s=25,
    zorder=2
)

ax.set_xlim(-9, 9)
ax.set_ylim(-9, 9)
ax.set_xlabel("MDYPL estimates")
ax.set_ylabel("rescaled MDYPL estimates")

ax.axline((0, 0), slope=1, color='grey', linestyle='--', linewidth=1, zorder=1)

ax.axline((0, 0), slope=1/mu_hat, color='grey', linestyle='-', linewidth=0.5, zorder=1)

legend_elements_features = [
    mpatches.Circle((0, 0), radius=5, facecolor='purple', edgecolor='purple', label='fou'),
    mpatches.Circle((0, 0), radius=5, facecolor='teal', edgecolor='teal', label='kar')
]
leg1 = ax.legend(
    handles=legend_elements_features, 
    loc='upper left', 
    title="Features",
    framealpha=1.0,
    edgecolor='black'
)
ax.add_artist(leg1)

legend_elements_slope = [
    plt.Line2D([0], [0], color='grey', linestyle='--', linewidth=1, label='1'),
    plt.Line2D([0], [0], color='grey', linestyle='-', linewidth=0.5, label=r'$1/\hat{\mu}$')
]
ax.legend(
    handles=legend_elements_slope, 
    loc='upper left', 
    bbox_to_anchor=(0.23, 1.0), 
    title="Slope",
    framealpha=1.0,
    edgecolor='black'
)

plt.show()