# LOTR Battle Predictor

This project implements the random forest machine learning method to predict the outcomes of battles between a Lord of the Rings hero and enemy.

### Dataset

The model is trained on `lotr_battles_1000.csv`, which contains 1,000 simulated battle records with the following columns:

| Column | Description |
|---|---|
| `HeroName` | Name of the hero |
| `HeroStrength` | Hero's strength stat |
| `HeroAgility` | Hero's agility stat |
| `HeroWeaponName` | Weapon used |
| `HeroWeaponBonus` | Numeric bonus granted by the weapon |
| `EnemyName` | Type of enemy |
| `EnemyStrength` | Enemy's strength stat |
| `EnemyAgility` | Enemy's agility stat |
| `EnemyNumber` | Number of enemies faced |
| `Outcome` | `1` = Victory, `0` = Defeat |

### Feature Engineering

The model uses several derived features:

```
HeroPower        = HeroStrength × 0.6 + HeroAgility × 0.4 + WeaponBonus
EnemyPower       = EnemyStrength × 0.6 + EnemyAgility × 0.4
EnemyTotalThreat = EnemyPower × EnemyNumber
PowerDelta       = HeroPower − EnemyTotalThreat
```

### Model

- **Algorithm:** Random Forest Classifier (200 trees)
- **Test Accuracy:** ~84%
- **Split:** 80/20 train/test with stratification

---
