# Classification de Places de Stationnement — CNN & Transfer Learning (PyTorch)

Projet de **vision par ordinateur** visant à classifier des images de places de stationnement en 3 catégories : **Libre**, **Occupé** et **Interdit**. Le projet explore plusieurs approches : un CNN personnalisé, l'augmentation de données, le transfer learning et le fine-tuning, tous implémentés avec **PyTorch** sur Google Colab (GPU).

---

## Objectif

Comparer les performances de plusieurs stratégies d'entraînement sur un dataset d'images de places de stationnement, afin d'identifier le meilleur compromis entre précision et temps d'entraînement.

---

## Structure des données

Les images sont organisées en classes dans un dossier Google Drive :

```
images_V2/
├── Libre/
├── Occupe/
├── Interdit/
├── train/        ← généré automatiquement (80%)
└── test/         ← généré automatiquement (20%)
```

- **Total** : 1 165 images (`.jpg` / `.jpeg`)
- **Entraînement** : 931 images
- **Test** : 234 images → scindé en 152 test / 82 validation

La division train/test est réalisée automatiquement avec un ratio de **80/20**, puis le jeu de test est subdivisé en **65% test / 35% validation** (pour conserver un maximum d'images à l'entraînement).

---

## Architecture du projet

### 1. Prétraitement de base (`data_transform`)
- Redimensionnement à **224×224** px (choix justifié par des tests à 64×64 et 128×128)
- Conversion en tenseur
- Normalisation avec les statistiques calculées sur le dataset d'entraînement :
  - `mean = [0.472, 0.465, 0.450]`
  - `std  = [0.173, 0.169, 0.170]`

### 2. Modèle CNN personnalisé — `TinyVGG`

Architecture inspirée de VGG, composée de 3 blocs convolutifs suivis d'un classifieur :

| Bloc | Couches |
|---|---|
| Bloc 1 | Conv2d(3→32) + ReLU × 2 + MaxPool |
| Bloc 2 | Conv2d(32→64) + ReLU × 2 + MaxPool |
| Bloc 3 | Conv2d(64→128) + ReLU × 2 + MaxPool |
| Classifieur | Dropout(0.5) + Linear(100352→512) + ReLU + Dropout(0.5) + Linear(512→3) |

Le choix des hyperparamètres (largeur, profondeur) a été guidé par des observations avec `torchinfo` : un modèle trop grand ou trop petit ne généralise pas bien.

### 3. Entraînement

- **Optimizer** : Adam (`lr=0.0001`, choisi après tests avec `0.1` et `0.001`)
- **Loss** : CrossEntropyLoss
- **Epochs** : 20 max
- **Early Stopping** : patience configurable, avec sauvegarde automatique du meilleur modèle (`best_model.pt`)
- **Batch size** : 32 (compromis durée/efficacité)
- **Reproductibilité** : seed fixé à 42

---

## Expériences & Résultats

### Modèle 0 — CNN Simple (sans augmentation)
- Meilleure précision de validation : **75.61%**
- Précision sur le test : **72.37%**
- Observations : confusions fréquentes entre les classes Libre et Interdit

### Modèle 1 — Augmentation 1 (`data_transform1`)
Transformations appliquées :
- `RandomHorizontalFlip(p=0.5)`
- `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1)`
- `RandomRotation(degrees=5)`

Dataset doublé par concaténation (`ConcatDataset`).

- Meilleure précision de validation : **78%**
- Précision sur le test : **86.84%** 
- Observations : forte amélioration sur toutes les classes, très bonne généralisation

### Modèle 2 — Augmentation 2 (`data_transform2`)
Transformations appliquées :
- `Resize(256)` + `CenterCrop(224)`
- `RandomHorizontalFlip(p=0.3)`
- `RandAugment(num_ops=4, magnitude=7)`

- Meilleure précision de validation : **73.17%**
- Précision sur le test : > 72% (meilleure que le baseline)
- Observations : amélioration de la classe Interdit mais pénalise légèrement les autres classes. Léger début de surapprentissage détecté.

### Modèle 3 — Transfer Learning (GoogLeNet)
- Poids pré-entraînés ImageNet
- Toutes les couches gelées sauf la couche finale (`fc`)
- Normalisation ImageNet : `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- Précision sur le test : **80%**

### Modèle 4 — Fine-Tuning (GoogLeNet)
- Dégel de la couche finale **et** du bloc `inception5b`
- Précision sur le test : **82%**
- Meilleur équilibre entre les classes Libre et Interdit par rapport au transfer learning pur

---

## Comparaison des modèles

| Modèle | Précision test | Époques | Données entraînement |
|---|---|---|---|
| SimpleCNN + Augmentation 1 | **86.84%** | ~13 | 1 862 (doublé) |
| Fine-Tuning GoogLeNet | 82% | ~20 | 931 |
| Transfer Learning GoogLeNet | 80% | ~25+ | 931 |
| SimpleCNN + Augmentation 2 | >72% | ~20 | 1 862 (doublé) |
| SimpleCNN Baseline | 72.37% | ~20 | 931 |

> **Modèle recommandé** : SimpleCNN + Augmentation 1, pour son excellente généralisation (86.84%) malgré un temps d'entraînement plus long dû au dataset doublé.

---

## Installation & Lancement

### Prérequis
- Compte Google (Google Colab + Google Drive)
- GPU activé dans Colab (`Exécution > Modifier le type d'exécution > GPU`)

### Dépendances Python

```bash
pip install torch torchvision torchinfo scikit-learn seaborn matplotlib pillow tqdm
```

### Structure attendue sur Google Drive

```
MyDrive/
└── Projet Pytorch/
    └── Projet 2/
        └── images_V2/
            ├── Libre/
            ├── Occupe/
            └── Interdit/
```

### Ordre d'exécution des cellules

Exécuter les cellules dans l'ordre. Les grandes étapes sont :
1. Montage du Drive et imports
2. Division des données (train/test)
3. Prétraitement et création des DataLoaders
4. Définition et entraînement du CNN simple
5. Augmentations de données (×2 expériences)
6. Transfer Learning et Fine-Tuning avec GoogLeNet
7. Comparaison des modèles

---

## Dépendances principales

| Bibliothèque | Usage |
|---|---|
| `torch` / `torchvision` | Modèles, couches, transformations |
| `torchinfo` | Visualisation des paramètres du modèle |
| `scikit-learn` | Matrice de confusion, rapport de classification |
| `matplotlib` / `seaborn` | Visualisations |
| `Pillow` | Chargement des images |
| `tqdm` | Barres de progression |

---

## Notes & Observations

- La résolution **224×224** est cruciale : à 64×64 l'accuracy chutait sous 30%, à 128×128 elle plafonnait à ~50%.
- L'**augmentation de données** n'améliore le modèle que si les transformations sont cohérentes avec la nature des images.
- Le **fine-tuning** surpasse le transfer learning pur : dégeler `inception5b` aide le modèle à mieux comprendre les images du domaine.
- **ResNet** (testé hors notebook) atteint des scores supérieurs en moins de 10 époques, ce qui s'explique par sa victoire à ImageNet.
- La question du surapprentissage reste ouverte : un modèle sans early stopping a parfois obtenu >90% en test, soulevant la question de savoir si l'overfitting est toujours négatif dans ce contexte.
