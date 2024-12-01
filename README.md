# Vehicle Detection Project

## Table of Contents

1. [Introduction](#introduction)
2. [Running the Scripts](#running-the-scripts)
3. [Project Structure](#project-structure)
4. [Contributors](#contributors)

---

## Introduction

The Vehicle Detection project aims to detect and count vehicles in images or video streams of roads using computer vision techniques. This project leverages various algorithms to accurately detect vehicles in different environments and conditions.

## Running the Scripts

Please ensure that you have the necessary dependencies installed before running the scripts by executing the following command:

```shell
pip install -r requirements.txt
```

0. **vehicule_detector.py** 
    - This script is for running the entire vehicle detection pipeline.
1. **tests** 
    - This script is for running the test suite.
    - Firt you will need to install your project in editable state
```shell
pip install -e .
```

## Project Structure

```
vehicule_detection/
│
├── src/  # Dossier contenant le code source du projet
│   ├── vehicule_detection/
│   │   ├── __init__.py  # Fichier d'initialisation du module
│   │   ├── vehicule_detector.py  # Script principal pour la détection de véhicules
│   │   └── utils.py  # Fonctions utilitaires
│   └── exploratory.ipynb  # NoteBook detailling step by step process
│
├── data/  # Dossier pour les données brutes et traitées
│   ├── raw/  # Données brutes
│   └── processed/  # Données traitées
│
├── tests/  # Dossier contenant les scripts de test
│   ├── __init__.py  # Fichier d'initialisation des tests
│   ├── test_vehicule_detector.py  # Tests pour le script de détection de véhicules
│   └── test_utils.py  # Tests pour les fonctions utilitaires
│
├── .gitignore  # Fichier pour ignorer certains fichiers dans Git
├── README.md  # Ce fichier
├── pyproject.toml # setup file to build project ass package
└── requirements.txt  # Dépendances Python
```

## Contributors

- [Zaghouane Fares](https://github.com/faresZzz)