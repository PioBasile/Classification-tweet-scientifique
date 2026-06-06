# Classification de tweets scientifiques — SciTweets

Projet de Machine Learning — M1 Génie Logiciel 2025–2026  
**Dorian Rey · Francesco-Pio Basile · Nathan Boureux**

---

## Objectif

Classification automatique de tweets issus du dataset [SciTweets](https://dl.acm.org/doi/10.1145/3511808.3557462) (Hafid et al., CIKM 2022) selon trois tâches de complexité croissante :

| Tâche | Description | Données |
|---|---|---|
| T1 | `SCI` vs `NON-SCI` | 1 140 tweets |
| T2 | `{CLAIM, REF}` vs `{CONTEXT}` | 375 tweets SCI |
| T3 | `CLAIM` × `REF` × `CONTEXT` (multi-label) | 375 tweets SCI |

---

## Méthodologie

Pipeline encapsulé dans un `ImbPipeline` :

```
Texte brut → Vectoriseur → Sampleur → Classifieur
```

Hyperparamètres optimisés automatiquement par **Optuna** (150 trials par tâche, CV 5-folds, F1 macro), sans fuite de données.

**Espace de recherche :**
- Vectoriseur : `CountVectorizer` / `TfidfVectorizer` + ngram, min_df, max_df
- Sampleur : `RandomOverSampler` / `RandomUnderSampler` / aucun
- Classifieur : LogReg, LinearSVM, Random Forest, Naive Bayes, KNN, Decision Tree

---

## Résultats

| Tâche | Meilleur modèle | F1 macro (CV) |
|---|---|---|
| T1 — SCI vs NON-SCI | LogReg | 0.769 |
| T2 — CLAIM+REF vs CONTEXT | Decision Tree | 0.653 |
| T3 — Multi-label | Random Forest | 0.825 |

**T3 — Métriques multi-label :**
- Hamming loss : 0.242
- Subset accuracy (exact match) : 0.493

---

## Structure du projet

```
├── ClassificationTweetScientifique.ipynb   # Notebook principal
├── scitweets_export.tsv                    # Dataset (à placer ici ou sur Drive)
└── README.md
```

---

## Lancer le notebook

### Google Colab
1. Ouvrir le notebook dans Colab
2. Monter Google Drive et adapter le chemin du dataset :
```python
from google.colab import drive
drive.mount('/content/drive')
path = '/content/drive/MyDrive/ColabNotebooks/scitweets_export.tsv'
```
3. Exécuter toutes les cellules

### Local
```bash
pip install optuna imbalanced-learn scikit-learn pandas numpy matplotlib seaborn wordcloud nltk
jupyter notebook ClassificationTweetScientifique.ipynb
```

---

## Dépendances principales

```
scikit-learn
imbalanced-learn
optuna
pandas
numpy
matplotlib
seaborn
nltk
wordcloud
```

---

## Dataset

SciTweets — collecté par le LIRMM et GESIS (Cologne) en 2022.

> Hafid, S., Schellhammer, S., Bringay, S., Todorov, K., & Dietze, S. (2022).
> *SciTweets — A Dataset and Annotation Framework for Detecting Scientific Online Discourse.*
> Proceedings of the 31st ACM CIKM, pp. 3988–3992.

---
