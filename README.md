Étape 1 : Préparation de l'environnement
Vérification de la disponibilité de PyTorch et de l'interopérabilité avec Transformers.

Validation du chargement automatique des poids et des tokeniseurs depuis le Hugging Face Hub.

Étape 2 : Classification d'opinion (Analyse des sentiments)
Modèle : distilbert/distilbert-base-uncased-finetuned-sst-2-english

Méthode : Application du pipeline en mode unitaire puis par lot sur une sélection d'avis positifs et négatifs.

Résultat : Le modèle atteint 80 % d'exactitude sur le lot de test. Les phrases directes sont classées avec une confiance proche de 1.00. En revanche, le modèle échoue sur les formulations sarcastiques (par exemple, "Wow, just what I wanted... another problem." est faussement prédit comme POSITIVE avec un score de 0.986), illustrant son incapacité à capter l'inversion d'opinion sans modélisation du contexte pragmatique.

Étape 3 : Classement Zero-Shot
Modèle : MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

Méthode : Formulation de la tâche comme un problème d'inférence en langage naturel (NLI). Le texte sert de prémisse et chaque étiquette candidate sert à formuler une hypothèse.

Constat : Pour une phrase sportive ( "L'équipe nationale de football a gagné la coupe hier." ), le label sports recueille plus de 99 % de la distribution de probabilité. L'ordre des classes déclarées n'impacte pas le résultat, mais une légère variation du libellé lexical (ex : sport news vs sports ) modifie considérablement le score de confiance final.

Étape 4 : Inférence Question–Réponse (QNLI)
Modèle : cross-encoder/qnli-electra-base

Méthode : Évaluation de paires composées d'une question et d'un passage source pour déterminer si l'information requise est contenue dans le texte.

Format des étiquettes : Selon la convention du benchmark GLUE/QNLI, LABEL_0correspondent à l'implication ( entailment , réponse présente) et LABEL_1à l'absence de réponse ( not entailment ). Le modèle filtre efficacement les paires non pertinentes sans nécessiter d'extraction de segment.

Étape 5 : Détection de paraphrases (QQP)
Modèle : textattack/bert-base-uncased-QQP

Méthode : Classification binaire de paires de questions issues de Quora Question Pairs.

Sorties : LABEL_1 indique une duplication sémantique et LABEL_0indique deux intentions distinctes. Le modèle discrimine correctement les paires aux vocabulaires quasi-identiques mais aux sens divergents, ainsi que les phrases aux formulations différentes ciblant le même objectif d'apprentissage.

Étape 6 : Vérification de l'acceptabilité grammaticale (CoLA)
Modèle : textattack/distilbert-base-uncased-CoLA

Méthode : Évaluation syntaxique sur des phrases conformes et corrompues (altérations d'ordre des mots et conjugaisons erronées).

Observation fondamentale : Le modèle attribué LABEL_1à la phrase canonique de Chomsky ( « Les idées vertes incolores dorment furieusement. » ), confirmant qu'il juge la légalité structurelle de la syntaxe anglaise modifiant de toute cohérence sémantique.

Analyse critique et discussion (Étape 7)
Limites constatées
Absence de contextualisation en analyse des sentiments : La tokenisation et la classification globale sans mémoire discursive rendent les modèles vulnérables au second degré, aux doubles négations et au sarcasme.

Sensibilité aux libellés en zero-shot : La probabilité attribuée dépend étroitement de la distance sémantique entre le mot-clé choisi et le texte. Un choix de synonyme inadéquat peut dégrader fortement la performance.

Opacité des réponses binaires : QNLI et QQP renvoient une décision globale sans localisation explicite des indices textuels déterminants.

Insensibilité sémantique de CoLA : Un texte grammaticalement parfait mais totalement absurde ou mensonger est systématiquement validé.

Pistes d'amélioration
Déployer des modèles multilingues (tels que xlm-roberta-baseou flaubert) pour dépasser la contrainte exclusive de la langue anglaise des modèles testés.

Calibrer le seuil de décision probabiliste sur QQP au lieu de conserver un simple seuil standard à 0.5, afin de moduler le compromis précision/rappel selon les besoins de l'application métier.

Réaliser un affinement supervisé ( fine-tuning ) des têtes de classification sur des données annotées propres au domaine cible pour fiabiliser la détection sur du vocabulaire spécialisé.




https://github.com/user-attachments/assets/462d1217-8982-4ce3-9ef2-ccc81a4046d7

