import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 110
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

RANDOM_STATE = 42
print('Setup OK')

COLUMN_NAMES = [
    'statut_compte', 'duree_mois', 'historique_credit', 'objet_credit',
    'montant_credit', 'epargne', 'anciennete_emploi', 'taux_versement',
    'statut_civil_sexe', 'autres_debiteurs', 'anciennete_residence',
    'propriete', 'age', 'autres_credits', 'logement', 'nb_credits',
    'emploi', 'nb_personnes_charge', 'telephone', 'travailleur_etranger',
    'defaut'
]

df = pd.read_csv(
    'data/raw/german.data',
    sep=' ',
    header=None,
    names=COLUMN_NAMES
)

df['defaut'] = (df['defaut'] == 2).astype(int)

print(f'Dataset chargé : {df.shape[0]} lignes × {df.shape[1]} colonnes')
print(f'Taux de défaut : {df["defaut"].mean():.1%}')
print(df.head())

info = pd.DataFrame({
    'dtype': df.dtypes,
    'valeurs_manquantes': df.isnull().sum(),
    'pct_manquant': (df.isnull().sum() / len(df) * 100).round(2),
    'nb_uniques': df.nunique()
})
print(info.to_string())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
counts = df['defaut'].value_counts()
colors = ['#5DCAA5', '#E24B4A']
axes[0].bar(['Bon payeur (0)', 'Défaut (1)'], counts.values, color=colors, width=0.5)
axes[0].set_title('Distribution de la cible', fontweight='bold')
axes[0].set_ylabel('Nombre de clients')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, f'{v} ({v/len(df):.0%})', ha='center', fontsize=11)
axes[1].pie(counts.values, labels=['Bon payeur', 'Défaut'],
            colors=colors, autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11})
axes[1].set_title('Proportions', fontweight='bold')
plt.tight_layout()
plt.savefig('data/processed/fig_target_distribution.png', bbox_inches='tight')
plt.show()
print(f'Ratio déséquilibre : {counts[0]/counts[1]:.1f}:1')

NUM_COLS = ['duree_mois', 'montant_credit', 'taux_versement',
            'anciennete_residence', 'age', 'nb_credits', 'nb_personnes_charge']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(NUM_COLS):
    for val, color, label in [(0, '#5DCAA5', 'Bon payeur'), (1, '#E24B4A', 'Défaut')]:
        axes[i].hist(df[df['defaut']==val][col], bins=20, alpha=0.6,
                     color=color, label=label, density=True)
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].legend(fontsize=8)
axes[-1].set_visible(False)
plt.suptitle('Distribution des variables numériques par classe', fontweight='bold')
plt.tight_layout()
plt.savefig('data/processed/fig_numeric_distributions.png', bbox_inches='tight')
plt.show()

top_cats = ['statut_compte', 'historique_credit', 'epargne',
            'anciennete_emploi', 'objet_credit', 'emploi']
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(top_cats):
    taux = df.groupby(col)['defaut'].mean().sort_values(ascending=False)
    axes[i].bar(range(len(taux)), taux.values,
                color=plt.cm.RdYlGn_r(taux.values))
    axes[i].set_xticks(range(len(taux)))
    axes[i].set_xticklabels(taux.index, rotation=30, ha='right', fontsize=9)
    axes[i].set_title(f'Taux défaut — {col}', fontsize=10, fontweight='bold')
    axes[i].axhline(df['defaut'].mean(), color='navy', linestyle='--',
                    alpha=0.5, label=f'Moy. ({df["defaut"].mean():.0%})')
    axes[i].legend(fontsize=8)
plt.suptitle('Taux de défaut par modalité', fontweight='bold')
plt.tight_layout()
plt.savefig('data/processed/fig_categorical_rates.png', bbox_inches='tight')
plt.show()

corr_df = df[NUM_COLS + ['defaut']].corr()
mask = np.triu(np.ones_like(corr_df, dtype=bool))
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={'size': 10}, linewidths=0.5)
ax.set_title('Matrice de corrélation', fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('data/processed/fig_correlation_matrix.png', bbox_inches='tight')
plt.show()
print('\nCorrélations avec la cible :')
print(corr_df['defaut'].drop('defaut').sort_values(ascending=False).to_string())