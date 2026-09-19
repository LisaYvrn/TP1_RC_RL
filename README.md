# TP1_RC_RL

# Courbes du TP1 : circuits RC et RL

## Commencer

Le fichier `courbes_tp1.py` est autonome. Le notebook `Courbes_TP1.ipynb`
contient le même code, découpé en cellules. Choisir l'un des deux.
Les seules bibliothèques externes nécessaires sont **NumPy** et **Matplotlib**.
Python 3.10 ou ultérieur est requis par les annotations du script.

```bash
python -m pip install numpy matplotlib
```

Pour voir immédiatement des résultats avec les mesures de l'an dernier :

```bash
python courbes_tp1.py --exemple-2025
```

Ce mode utilise uniquement les valeurs des tableaux de `TP_Laura.pdf`,
page 13, figure 6 (RC), et page 30, figure 16 (RL). Il ne les présente
jamais comme des mesures personnelles. Tous les graphiques issus de ces
valeurs portent la mention **EXEMPLE 2025** et vont dans un dossier séparé.
Les valeurs de la colonne de temps moyen sont retranscrites telles qu'elles
sont publiées, sans recalculer les moyennes à partir des bornes arrondies.

Pour traiter ses propres mesures, compléter `RC` et `RL` en début de
script, conserver `EXEMPLE_2025 = False`, puis exécuter :

```bash
python courbes_tp1.py
```

Dans le notebook, modifier la cellule de configuration puis exécuter
les cellules dans l'ordre. Pour l'essai avec les anciennes données,
mettre `EXEMPLE_2025 = True` dans cette cellule. Dans un éditeur Python,
on peut aussi simplement ouvrir le fichier `.py` et l'exécuter.

**Avant toute saisie**, les listes sont vides (`np.nan`). Le programme
produit alors les huit illustrations théoriques, mais aucune régression
expérimentale. Il ne crée pas de mesures fictives pour combler les trous.

## Renseigner les mesures

Les quatre colonnes de chaque circuit sont :

| Clé | Grandeur | Unité de saisie |
|---|---|---|
| `R_ohm` | Résistance externe R | ohm |
| `dR_ohm` | Incertitude absolue sur R | ohm |
| `tau_us` | Constante de temps mesurée | microseconde |
| `dtau_us` | Incertitude absolue sur cette durée | microseconde |

Une même position dans les quatre listes correspond à une même mesure.
**Saisir `105`, et non `105e-6`, pour 105 microsecondes.** La conversion
en secondes est faite automatiquement avant les calculs.
Ne pas utiliser zéro pour représenter une mesure absente. Une ligne
entièrement `np.nan` est ignorée ; une ligne partiellement complétée
est signalée comme une erreur de saisie.

Le script demande trois points pour produire un ajustement. Cela ne remplace
pas les consignes : le sujet impose au minimum une dizaine de valeurs de
R pour le RC. Le nombre limité de points RL dans l'exemple 2025 est une
limitation de cet exemple, pas une recommandation pour votre manipulation.

Les incertitudes sont absolues : saisir par exemple l'incertitude en ohms,
pas directement un pourcentage. La précision du matériel utilisé doit être
relevée le jour du TP ; elle n'est pas copiée du matériel de 2025.
Les références constructeur sont facultatives :
`C_REFERENCE_UF = (valeur, incertitude)` et
`L_REFERENCE_MH = (valeur, incertitude)`.
Leurs unités sont respectivement le microfarad et le millihenry.

La fonction `depuis_encadrements(R_ohm, dR_ohm, tau_min_us, tau_max_us)`
constitue une autre entrée possible. Elle calcule la moyenne des bornes
et la demi-largeur de l'intervalle. Les bornes doivent encadrer une **durée**,
et non simplement la date absolue d'un curseur. Si l'origine est elle-même
incertaine, cette contribution doit être intégrée à l'encadrement.

## Graphiques produits

| Nom de fichier, sans extension | Contenu | Rôle |
|---|---|---|
| `rc_regression` | tau en fonction de R et droite affine | Régression demandée |
| `rl_regression` | 1/tau en fonction de R et droite affine | Régression demandée |
| `rc_C_apparent` | C apparent = tau/R, avec comparaison à la pente | Complément pour la discussion |
| `rl_L_apparent` | L apparent = R tau, avec comparaison à la régression | Complément pour la discussion |
| `rl_tau_R` | tau(R) et courbe déduite de l'ajustement de 1/tau | Complément, pas un second ajustement |
| `rc_residus`, `rl_residus` | Écarts entre mesures et droite ajustée | Diagnostic visuel |
| `rc_echelon_theorique`, `rl_echelon_theorique` | Réponse à un échelon et seuil à t = tau | Illustration du modèle |
| `rc_integrateur_*_theorie` | Créneau, triangle et sinus en entrée | Illustration du cas limite RC |
| `rl_derivateur_*_theorie` | Créneau, triangle et sinus en entrée | Illustration du cas limite RL |

Avec les deux séries complétées et les options par défaut, cela donne
**15 figures**, chacune en PDF et PNG. Les fichiers CSV de valeurs calculées,
le `bilan_numerique.txt` et le `resultats.json` sont également enregistrés.
Le JSON conserve les paramètres en unités SI sans arrondi d'affichage ;
le bilan arrondit valeur et incertitude au même rang.

Les acquisitions d'oscilloscope, si elles sont ajoutées, produisent des
figures supplémentaires. Les mesures sont représentées par des points avec
barres horizontales et verticales ; les ajustements sont des lignes.
Pour ne conserver que les deux régressions parmi les figures d'exploitation,
mettre `TRACER_COMPLEMENTS = False`. Les tableaux restent calculés.

## Modèles et incertitudes

Le modèle affine est toujours `y = a R + b`, avec une **ordonnée libre**.
Tous les paramètres ci-dessous sont exprimés en unités SI :

- RC : y = tau, a = C, b = R' C ; donc C = a et R' = b/a.
- RL : y = 1/tau, a = 1/L, b = R'/L ; donc L = 1/a et R' = b/a.

L'incertitude de l'ordonnée du RL est **Delta(1/tau) = Delta tau / tau²**.
On ne conserve donc pas les barres Delta tau sur le graphique de 1/tau.
Les grandeurs dites apparentes n'intègrent pas R' : elles servent justement
à étudier les limites d'une détermination directe.

Le calcul des coefficients utilise les **moindres carrés non pondérés**.
L'incertitude sur R n'intervient pas dans la minimisation centrale, mais
elle intervient dans la propagation des incertitudes des coefficients.
Ce n'est pas une régression orthogonale ou une régression pondérée.
Lorsque les incertitudes verticales sont très différentes ou les
incertitudes horizontales importantes, l'adéquation de ce choix doit être
réexaminée et justifiée dans le rapport. Le script affiche des avertissements.

Le sujet impose une propagation additive au premier ordre. Le script
l'applique aux expressions des coefficients OLS, comme dans l'annexe du
compte rendu LaTeX déjà préparé. En posant X_i = x_i - moyenne(x),
Y_i = y_i - moyenne(y), Sxx = somme(X_i²), on utilise :

```text
da/dy_i = X_i / Sxx
da/dx_i = (Y_i - 2 a X_i) / Sxx
db/dy_i = 1/n - moyenne(x) da/dy_i
db/dx_i = -a/n - moyenne(x) da/dx_i

Delta a = somme(|da/dx_i| Delta x_i + |da/dy_i| Delta y_i)
Delta b = somme(|db/dx_i| Delta x_i + |db/dy_i| Delta y_i)

Delta C = Delta a                         (RC)
Delta L = Delta a / a^2                   (RL)
Delta R' = Delta b/|a| + |b| Delta a/a^2   (les deux)
```

Ce sont des incertitudes propagées, **pas des écarts-types ni des intervalles
à 95 %**. Pour b/a, la propagation est prudente : elle ne tire pas parti de
la corrélation entre a et b. Les incertitudes obtenues ne sont donc pas
nécessairement celles publiées dans les anciens rapports, qui emploient
notamment des droites extrêmes. Ce choix méthodologique est explicite,
pas une reproduction de leurs calculs d'incertitude.

Les résidus sont r_i = y_i - (a R_i + b). Leurs barres indiquent l'échelle
Delta y_i + |a| Delta R_i, pour comparer visuellement les écarts aux
incertitudes des mesures. **Ce n'est ni l'incertitude statistique complète
sur chaque résidu ni un test d'acceptation du modèle.** Un R² proche de 1
ne suffit pas à conclure. Aucun point n'est supprimé automatiquement parce
qu'il s'écarte de la droite ; aucun R' négatif n'est remplacé par sa valeur absolue.

## Courbes théoriques et limites de leur interprétation

Les réponses à l'échelon reprennent les expressions de l'annexe du sujet.
Leurs axes sont sans dimension : t/tau et tension/E. Elles localisent
63,2 % de la montée RC et 36,8 % du niveau initial d'une décroissance
RL idéale.

Pour les cas limites, l'entrée est **centrée**, de période T et d'amplitude E.
Les rapports tau/T = 10 et 0,01 sont des choix d'illustration modifiables,
pas des constantes mesurées. Les échelles verticales distinctes sont
indiquées explicitement : l'entrée se lit à gauche et la sortie à droite.
Il ne faut pas comparer les hauteurs graphiques sans lire ces deux axes.
L'approximation intégrale ou dérivée est superposée à la sortie quand
elle est représentable comme une fonction ordinaire.

Ces simulations sont une mise en œuvre numérique du modèle du premier ordre,
et non des courbes expérimentales extraites des documents. Elles utilisent
une entrée maintenue constante pendant de petits pas et un état initial
périodique calculé, sans transitoire de démarrage artificiel.
Pour un créneau discontinu en RL, on trace la réponse du modèle avec ses
impulsions aux fronts ; on ne remplace pas la dérivée par une fausse courbe
nulle partout. Le fait que la sortie soit faible entre deux fronts ne
suffit pas, à lui seul, à vérifier le caractère dérivateur.

Le RL illustré mesure la tension de la partie inductive **idéale**. La
mesure aux bornes d'une bobine réelle peut présenter un plateau non nul.
Pour les acquisitions, le seuil de pointage proposé utilise donc les
niveaux initial et final réellement observés :
`U_tau = U_final + (U_initial - U_final) exp(-1)`.
Cette forme généralise l'exponentielle ; le script ne choisit pas à votre
place les plateaux ou le temps de départ.

## Ajouter une acquisition d'oscilloscope

Le programme ne connaît pas le format exact de votre appareil. Exporter
les données puis indiquer leur format dans `ACQUISITIONS`. Par exemple,
pour un fichier avec une ligne d'en-tête puis les trois colonnes
`temps_s;CH1_V;CH2_V` :

```python
ACQUISITIONS = [{
    "fichier": "mesures/rc_echelon.csv",
    "nom": "rc_63",
    "circuit": "RC",
    "titre": "Charge du condensateur - acquisition",
    "separateur": ";",
    "lignes_entete": 1,
    "colonnes": (0, 1, 2),
    "unite_temps": "s",
    "double_axe": False,
    "pointage": {
        "t0_s": None,
        "tau_s": None,
        "u_initial_V": None,
        "u_final_V": None,
    },
}]
```

Les colonnes se comptent à partir de zéro. Les formats temporels reconnus
sont `s`, `ms`, `us`. Les valeurs du bloc `pointage` sont en **secondes et
volts**, quelle que soit l'unité de temps du fichier ; les laisser toutes
à `None` désactive l'annotation. Elles doivent décrire un seul transitoire.
Si la courbe n'a pas atteint son plateau avant le front suivant, ne pas
prendre arbitrairement le dernier échantillon pour sa limite asymptotique.

Les virgules décimales sont acceptées avec un séparateur point-virgule.
Les fichiers sont lus en UTF-8 ; exporter ou convertir l'encodage si
l'appareil utilise un autre format. Aucune mesure n'est lissée ou ajustée
automatiquement. Les courbes relient les échantillons enregistrés pour
restituer la trace. Les incertitudes instrumentales des échantillons ne
sont pas inventées ; les barres des régressions viennent de vos propres
Delta R et Delta tau. Conserver les acquisitions et captures originales.

La même liste permet d'ajouter les trois formes d'entrée pour chaque cas
limite, avec les noms suggérés dans le script. Les vraies acquisitions
sont ignorées pendant le mode `EXEMPLE_2025` pour éviter tout mélange.

## Insérer les figures dans le rapport LaTeX préparé

Les noms `rc_regression.pdf` et `rl_regression.pdf` correspondent aux noms
attendus dans le rapport. Copier les deux fichiers produits depuis
`resultats_TP1/mesures/` vers le dossier `figures/` du projet LaTeX.
Les acquisitions nommées `rc_63`, `rl_37`, `rc_integrateur_creneau`, etc.
produisent aussi les PNG attendus par les emplacements du rapport.

Les fichiers du dossier `theorie/` ne doivent **pas remplacer des captures
d'oscilloscope en étant présentés comme des observations**. Ils peuvent
compléter la partie théorique, avec une légende explicite. Les résidus
sont volontairement enregistrés dans des figures séparées. Il n'est pas
nécessaire d'insérer toutes les figures dans les 30 pages : choisir celles
qui appuient la discussion, et citer chaque figure dans le texte.

## Rangement et contrôles

```text
resultats_TP1/
    mesures/          # vos regressions, complements et tableaux
    acquisitions/     # vos exports oscilloscope retraces
    theorie/          # illustrations explicitement identifiees
    bilan_numerique.txt
    resultats.json
    EXEMPLE_2025/     # dossier independant lors d'un essai avec les anciens TP
```

Les noms sont réutilisés à l'exécution suivante : les fichiers régénérés
sont remplacés. Les autres ne sont pas supprimés automatiquement.
Consulter la liste des fichiers de **l'exécution courante** dans le bilan
pour ne pas utiliser un ancien résultat laissé dans le dossier après une
erreur ou une modification des options.

Pour enregistrer sans afficher les fenêtres :
`python courbes_tp1.py --sans-affichage`.
Pour supprimer les illustrations : ajouter `--sans-theorie`.
Pour changer le dossier : `--sortie chemin_du_dossier`.
Les erreurs de saisie sont signalées circuit par circuit ; une erreur
sur le RL n'empêche pas le traitement du RC.

## Sources et choix de réalisation

Documents fournis : sujet `TP 1-2-3 2026 electrocinetique MPCI.pdf`, partie
TP1, pages 1-7 ; `Guide daide pour la rédaction des comptes rendus.pdf`,
notamment pages 3-5 ; `TP1_DERRIENNIC_LELOUP.pdf`, graphique de tau(R)
page 11 et prévisions RL pages 16-17 ; `TP_Laura.pdf`, tableaux pages 13
et 30, régressions pages 15 et 31, comportements limites pages 19-21 et
34-35. Le premier ancien rapport ne contient pas de série de mesures RL.

Les modèles et la propagation additive viennent du sujet. Les figures des
anciens rapports servent de référence de présentation. L'algorithme de
propagation des coefficients, les simulations et les contrôles de saisie
sont les choix explicites de ce script ; ils ne sont pas présentés comme
le code original des étudiants.

Documentation des fonctions employées :
- NumPy, `numpy.linalg.lstsq` (moindres carrés).
- Matplotlib, `Axes.errorbar` (barres d'incertitude), `Figure.savefig`
  (enregistrement PDF et PNG).

La version effectivement utilisée de chaque bibliothèque est reportée
dans le bilan. Cette version du script a été exécutée et contrôlée
avec NumPy 2.3.5 et Matplotlib 3.10.8.
