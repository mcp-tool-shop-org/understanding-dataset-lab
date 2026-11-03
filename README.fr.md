<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.md">English</a> | <a href="README.hi.md">हिन्दी</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/understanding-dataset-lab/readme.png" alt="understanding-dataset-lab" width="400">
</p>

<p align="center">
  <b>empathy</b>: "Identification with or understanding of the thoughts, feelings, or emotional state of another person."<br>
  <sub>Definition from <a href="https://en.wiktionary.org/wiki/empathy">Wiktionary</a>, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>.</sub><br><br>
  Cognitive empathy, as udlab means it: an accurate model of what each party in a situation knows, believes, wants and needs.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-prototype-orange" alt="Prototype">
  <a href="https://github.com/mcp-tool-shop-org/understanding-dataset-lab/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/understanding-dataset-lab/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/understanding-dataset-lab/"><img src="https://img.shields.io/badge/docs-landing_page-blue" alt="Landing page"></a>
</p>

> **Prototype.** udlab est un projet de recherche en cours et ouvert. La méthode et le code sont disponibles pour être consultés et exécutés. Jusqu’à présent, ses résultats consistent en des vérifications de la qualité de la génération de scènes, sur des bases de données de développement. Il n’y a pas de résultat final ou définitif, pas de modèle entraîné, et il n’est fait aucune affirmation quant à savoir si une IA est capable de comprendre qui que ce soit. Les étapes peuvent encore modifier la manière dont les scènes sont créées.

**understanding-dataset-lab (udlab) crée des scénarios de test pour l’empathie cognitive : déterminer si un modèle est capable de conserver une image précise de ce que chaque partie prenante d’une scène sait, croit, veut et a besoin.** Des scènes courtes avec plusieurs personnes sont planifiées par le biais d’un code, puis transformées en prose par un modèle d’écriture local. La réponse à chaque question concernant une scène provient du plan du code, et non d’un modèle.

Il s’agit du complément de [reasoning-dataset-lab](https://github.com/mcp-tool-shop-org/reasoning-dataset-lab), qui fait la même chose pour les énigmes et les éléments de vérification.

## Pourquoi des scènes avec des éléments définis par un code ?

Dans une scène planifiée, la réponse correcte découle mécaniquement des événements : qui s’est déplacé où, qui pouvait voir, qui a dit quoi et qui l’a entendu. Une personne qui n’était pas visible lorsqu’un objet a été déplacé croit toujours qu’il se trouve à l’endroit où elle l’a vu pour la dernière fois. Étant donné que ces règles sont intégrées dans le code, la clé est calculée, et non jugée.

Une *clé définie par un code* signifie :
- la réponse provient du plan de la scène établi par le code ;
- elle est scellée avec une clé privée ;
- elle n’est jamais écrite ou évaluée par un modèle.

Chaque version en prose est vérifiée par le code par rapport au plan avant d’être prise en compte.

## État : ce qui a été essayé

Les résultats sont des comptages effectués sur des bases de données de développement, portant sur 160 versions. Les résultats négatifs sont présentés en premier, car ils ont façonné la conception.

| Étape | Ce qui a été essayé | Résultat | Adopté ? |
|---|---|---|---|
| 3 | Une liste des formes de phrases acceptées dans l’invite de l’écrivain | 111/160 versions ont réussi, contre 127/160 pour l’invite de référence. L’écrivain n’a pas utilisé les formes répertoriées. | **Non**, en vertu de la règle préenregistrée |
| 4 | Formes de phrases attribuées par le code | 89/160, contre 120/160 pour la référence. La validation de développement a échoué, il n’y a donc pas eu d’exécution de confirmation. 16 versions par branche (10 %) n’ont jamais été générées, car l’invite et le budget de sortie ont dépassé la limite de contexte de l’écrivain. | **No** |
| 5 | Déplacements du code et discours réécrits (révision 2) | L’exécution de développement a réussi toutes les 7 validations préenregistrées. Premières ébauches : 126/160, contre un minimum de 112 ; 132/160 après le pipeline. Les échanges de code ont permis de conserver 2,501/2,501. Les réécritures de discours ont permis de conserver 350/358. 0 rendu n’a dépassé la limite de contexte. | A réussi ses validations de développement |

**Pas encore démontré :**
- aucune exécution de confirmation scellée de l’étape 5 ;
- aucun modèle entraîné sur les données udlab ;
- rien pour l’instant sur le fait qu’un modèle comprenne les personnes dans une scène.

## Ce qui est proposé ici

- **`udlab-gen`**, le générateur. Il planifie les scénarios, exécute le pipeline de l’étape 5, scelle les clés de chaque ensemble avec une clé privée et vérifie un ensemble écrit en le rendant à nouveau et en comparant les deux versions.
- **Sondes de raccourci**, exécutées en dehors du processus principal. Elles testent si les caractéristiques superficielles d’une scène révèlent la réponse.
- **L’invite de l’écrivain** (`prompts/`). L’écrivain est `qwen3.6:35b`, exécuté localement via Ollama.
- **Un lanceur Colab** (`scripts/colab/`) pour exécuter l’écrivain sur un GPU loué.
- **759 tests.**

## Essayez-le sans GPU

L’écrivain par défaut est un modèle CPU, de sorte que tout le pipeline s’exécute sur n’importe quelle machine :

```bash
git clone https://github.com/mcp-tool-shop-org/understanding-dataset-lab
cd understanding-dataset-lab
uv run udlab-gen --count 3 --out-dir artifacts/demo
uv run udlab-gen --probe-only artifacts/demo/sealed_store/keys.sealed.jsonl
uv run pytest
```

- Python 3.15 ou version ultérieure ; `uv` l’installe si nécessaire.
- La première commande écrit une partie publique (`part_public/`) et un stockage scellé (`sealed_store/`) dans `artifacts/demo/`, un dossier que Git ignore.
- `uv run udlab-gen --help` répertorie toutes les options.

## Ce qui reste privé

Les éléments suivants restent privés, car leur publication révélerait les réponses :
- la base de données de test scellée et les bases de données de clés de développement ;
- tous les stockages scellés et les journaux de génération ;
- les enregistrements d’exécution avec leurs clés de plan.

Aucun de ces éléments ne se trouve dans ce dépôt ou dans son historique. Les tests utilisent uniquement des clés synthétiques et des scènes inventées.

## Sécurité et confiance

- **Données concernées :** fichiers dans le dossier `--out-dir` que vous choisissez, et l’invite de l’écrivain. Avec `--writer ollama`, il appelle également un serveur Ollama local.
- **Données non concernées :** rien en dehors du dossier de sortie. Il n’y a pas de télémétrie et aucune communication réseau, à l’exception du serveur Ollama local et, si vous l’exécutez, du lanceur Colab.
- **Autorisations :** aucune autorisation au-delà de la lecture et de l’écriture dans le dossier de sortie. En mode GPU, il arrête le serveur Ollama qu’il a démarré si la mémoire vidéo dépasse `--vram-halt-mib`.
- Signalez les problèmes de sécurité comme le décrit [SECURITY.md](SECURITY.md).

## Licence

Le code et le texte du dépôt sont soumis à la [licence MIT](LICENSE). La définition citée sous le logo provient de Wiktionnaire et reste sous [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Elle n’est pas utilisée dans un ensemble de données ou une invite.
