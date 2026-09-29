# -*- coding: utf-8 -*-
"""TP1 - Circuits RC et RL : figures et exploitation des mesures.

Dependances : numpy et matplotlib. Python >= 3.10.
Installation : python -m pip install numpy matplotlib
Execution   : python courbes_tp1.py
Test 2025   : python courbes_tp1.py --exemple-2025 --sans-affichage

Les mesures personnelles sont vides par defaut. Les donnees de l'exemple
proviennent de TP_Laura.pdf, pages 13 et 30, et ne sont PAS vos mesures.
Les simulations sont toujours identifiees et rangees a part.

La droite centrale est obtenue par moindres carres non ponderes. Pour les
incertitudes sur la pente et l'ordonnee a l'origine, le graphique ajoute les
deux droites extremes utilisees dans le TP : la droite la plus pentue et la
droite la moins pentue, construites a partir des barres d'erreur des premier
et dernier points. On utilise alors Delta a = (a_max-a_min)/2 et
Delta b = (b_max-b_min)/2. La propagation additive reste calculee en interne
comme controle, mais les resultats finaux utilisent les droites extremes.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# %% 1. ZONE A COMPLETER : aucune mesure personnelle n'est inventee.
# Une colonne = une grandeur ; un meme indice = une meme mesure.
# Resistance et incertitude en ohms ; tau et incertitude en MICROSECONDES.
# Remplacer les listes [np.nan] * 10 par vos listes de nombres.
RC = {
    "R_ohm":    [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000] ,
    "dR_ohm":   [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "tau_us":   [15.66, 55.80, 108, 213, 305, 395, 494.5, 605, 699, 796, 900, 988],
    "dtau_us":  [0.66, 3.2, 4.4, 10, 10, 12, 17.5, 21, 29, 36, 36, 28],
}
RL = {
    "R_ohm":    [100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
    "dR_ohm":   [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "tau_us":   [770, 190.5, 97.6, 50.6, 33.8, 26.15, 21.45, 18.30, 16.14, 14.8, 13.86, 13.22],
    "dtau_us":  [32, 7.5, 4.4, 2.4, 1.6, 0.65, 0.65, 0.58, 0.3, 0.4, 0.38, 0.3],
}

# Valeurs constructeur facultatives : (valeur, incertitude absolue).
# Laisser None tant qu'elles ne sont pas connues. Ne pas reprendre
# automatiquement les caracteristiques des composants de l'an dernier.
C_REFERENCE_UF = (None, None)    # capacite en microfarads
L_REFERENCE_MH = (None, None)    # inductance en millihenrys

EXEMPLE_2025 = False            # True = anciens resultats, jamais vos mesures
AFFICHER = True                # False = enregistrer sans ouvrir les figures
TRACER_THEORIE = True           # illustrations distinctes des acquisitions
TRACER_COMPLEMENTS = True       # C_app(R), L_app(R), tau_RL(R), residus
RACINE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DOSSIER_SORTIE = RACINE / "resultats_TP1"

# Illustrations sans dimension, PAS des constantes de temps mesurees.
TAU_SUR_T_INTEGRATEUR = 10.0     # tau / T >> 1
TAU_SUR_T_DERIVATEUR = 0.01      # tau / T << 1

# Exports oscilloscope facultatifs. Chaque fichier doit contenir les
# colonnes temps, CH1, CH2 (l'ordre est configurable). Voir LIRE_MOI.md.
ACQUISITIONS = []
# Exemple de configuration A COMPLETER, puis a placer dans la liste :
# {
#     "fichier": "mesures/rc_echelon.csv", "nom": "rc_63",
#     "circuit": "RC", "titre": "Charge du condensateur - acquisition",
#     "separateur": ";", "lignes_entete": 1, "colonnes": (0, 1, 2),
#     "unite_temps": "s", "double_axe": False,
#     "pointage": {"t0_s": None, "tau_s": None,
#                  "u_initial_V": None, "u_final_V": None},
# }
# Pour les cas limites, utiliser par exemple les noms suivants :
# rc_integrateur_creneau, rc_integrateur_triangle, rc_integrateur_sinus,
# rl_derivateur_creneau, rl_derivateur_triangle, rl_derivateur_sinus.

# %% 2. DONNEES, AJUSTEMENT ET INCERTITUDES
@dataclass
class Ajustement:
    a: float
    b: float
    da: float
    db: float
    r2: float
    residus: np.ndarray


@dataclass
class DroitesExtremes:
    a_min: float
    b_min: float
    a_max: float
    b_max: float
    da: float
    db: float
    points_min: tuple[tuple[float, float], tuple[float, float]]
    points_max: tuple[tuple[float, float], tuple[float, float]]


def avertir(message: str, journal: list[str]) -> None:
    ligne = "ATTENTION : " + message
    print(ligne)
    journal.append(ligne)


def depuis_encadrements(R_ohm, dR_ohm, tau_min_us, tau_max_us) -> dict:
    """Option : convertir des encadrements de DUREE en tau et Delta tau.

    Si les bornes sont des dates absolues de curseur, soustraire d'abord
    l'origine et tenir compte de son incertitude : ce ne sont pas des durees.
    """
    lo, hi = np.asarray(tau_min_us, float), np.asarray(tau_max_us, float)
    if lo.shape != hi.shape or np.any(hi < lo):
        raise ValueError("Encadrements incoherents : tau_max doit etre >= tau_min.")
    return {"R_ohm": R_ohm, "dR_ohm": dR_ohm,
            "tau_us": (lo + hi) / 2, "dtau_us": (hi - lo) / 2}


def donnees_2025() -> tuple[dict, dict]:
    """Transcription des colonnes R, Delta R, tau_moy, Delta tau publiees.

    Source : TP_Laura.pdf, Bouchard-Mourier / Torres, 15 septembre 2025,
    page 13 (RC, figure 6) et page 30 (RL, figure 16).
    On conserve les valeurs arrondies de tau_moy du tableau, sans les
    remplacer par de nouveaux milieux calcules a partir de bornes arrondies.
    """
    rc = {"R_ohm": [250, 500, 750, 1000, 1250, 1500, 2000, 2500, 3000, 3500],
          "dR_ohm": [5, 10, 15, 20, 25, 30, 40, 50, 60, 70],
          "tau_us": [28.9, 52.6, 80.0, 105, 130, 154, 206, 257, 304, 362],
          "dtau_us": [1.1, 1.0, 1.6, 2, 2, 2.5, 2.5, 2.5, 3, 3]}
    rl = {"R_ohm": [1000, 1500, 2000], "dR_ohm": [20, 30, 40],
          "tau_us": [98, 64, 50], "dtau_us": [2, 2, 2]}
    return rc, rl


def preparer_donnees(donnees: dict, circuit: str, journal: list[str]):
    """Valider la saisie, ignorer les lignes entierement vides, passer en SI."""
    cles = ("R_ohm", "dR_ohm", "tau_us", "dtau_us")
    colonnes = [np.asarray(donnees[k], dtype=float) for k in cles]
    if any(c.ndim != 1 for c in colonnes):
        raise ValueError(f"{circuit} : chaque colonne doit etre une liste simple.")
    if len({len(c) for c in colonnes}) != 1:
        raise ValueError(f"{circuit} : les quatre listes doivent avoir la meme longueur.")
    matrice = np.column_stack(colonnes)
    vide = np.isnan(matrice).all(axis=1)
    invalide = ~np.isfinite(matrice).all(axis=1) & ~vide
    if invalide.any():
        lignes = (np.flatnonzero(invalide) + 1).tolist()
        raise ValueError(f"{circuit} : lignes incompletes ou infinies : {lignes}.")
    matrice = matrice[~vide]
    if len(matrice) == 0:
        journal.append(f"{circuit} : aucune mesure renseignee ; aucun resultat invente.")
        print(journal[-1])
        return None
    if len(matrice) < 3:
        raise ValueError(f"{circuit} : renseigner au moins 3 mesures pour cet ajustement.")
    R, dR, tau, dtau = matrice.T.copy()
    if np.any(R < 0) or np.any(dR < 0) or np.any(tau <= 0) or np.any(dtau <= 0):
        raise ValueError(f"{circuit} : R et Delta R >= 0 ; tau et Delta tau > 0.")
    if np.unique(R).size < 2:
        raise ValueError(f"{circuit} : il faut au moins deux resistances distinctes.")
    if circuit == "RC" and np.unique(R).size < 10:
        avertir("RC : le sujet demande au minimum une dizaine de valeurs de R.", journal)
    if circuit == "RL" and len(R) < 5:
        avertir("RL : peu de points ; examiner particulierement l'ordonnee a l'origine.", journal)
    if circuit == "RL" and np.any(dtau >= tau):
        raise ValueError("RL : un intervalle de tau contient zero ; inversion non exploitable.")
    if np.any(dtau / tau > 0.10):
        avertir(f"{circuit} : Delta tau/tau > 10 % ; controler les approximations au premier ordre.", journal)
    if np.any(dR == 0):
        avertir(f"{circuit} : certaines Delta R sont nulles ; ce choix doit etre justifie.", journal)
    ordre = np.argsort(R, kind="stable")
    return R[ordre], dR[ordre], tau[ordre] * 1e-6, dtau[ordre] * 1e-6


def regression_additive(x, y, dx, dy) -> Ajustement:
    """y = a*x+b : OLS non pondere, ordonnee libre, propagation additive.

    La position centrale est une regression VERTICALE non ponderee.
    dx intervient dans les incertitudes propagees, pas dans la minimisation.
    La dispersion residuelle n'est pas ajoutee comme une nouvelle erreur.
    Un mauvais modele ne se corrige donc pas en gonflant automatiquement da.
    """
    x, y, dx, dy = [np.asarray(v, dtype=float) for v in (x, y, dx, dy)]
    if x.ndim != 1 or not (x.shape == y.shape == dx.shape == dy.shape) or len(x) < 3:
        raise ValueError("Regression : quatre vecteurs de meme taille, au moins 3 points.")
    if not all(np.isfinite(v).all() for v in (x, y, dx, dy)):
        raise ValueError("Regression : toutes les valeurs doivent etre finies.")
    if np.any(dx < 0) or np.any(dy < 0):
        raise ValueError("Regression : incertitudes negatives.")
    n = len(x)
    xm, ym = float(np.mean(x)), float(np.mean(y))
    X, Y = x - xm, y - ym
    Sxx = float(X @ X)
    if Sxx <= 0:
        raise ValueError("Regression impossible : toutes les abscisses sont identiques.")
    # Centrage et changement d'echelle pour le conditionnement numerique.
    echelle = np.sqrt(Sxx / n)
    A = np.column_stack((X / echelle, np.ones(n)))
    (alpha, beta), _, rang, _ = np.linalg.lstsq(A, Y, rcond=None)
    if rang != 2:
        raise ValueError("Regression : matrice de rang insuffisant.")
    a = float(alpha / echelle)
    b = float(ym + beta - a * xm)
    # Derivees EXACTES des coefficients OLS par rapport aux donnees.
    da_dy = X / Sxx
    da_dx = (Y - 2 * a * X) / Sxx
    db_dy = 1 / n - xm * da_dy
    db_dx = -a / n - xm * da_dx
    da = float(np.sum(np.abs(da_dx) * dx + np.abs(da_dy) * dy))
    db = float(np.sum(np.abs(db_dx) * dx + np.abs(db_dy) * dy))
    residus = y - (a * x + b)
    Syy = float(Y @ Y)
    r2 = float(1 - (residus @ residus) / Syy) if Syy > 0 else float("nan")
    return Ajustement(a, b, da, db, r2, residus)


def droites_extremes(x, y, dx, dy) -> DroitesExtremes:
    """Construire les deux droites extremes de la methode utilisee dans le TP.

    Les points doivent etre classes par abscisse croissante. Pour une tendance
    croissante y=f(x) :
      - pente maximale : premier point en bas-a-droite de sa barre d'erreur et
        dernier point en haut-a-gauche ;
      - pente minimale : premier point en haut-a-gauche et dernier point en
        bas-a-droite.

    Cette construction est celle decrite dans le compte rendu de reference.
    Elle est surtout pertinente lorsque les premier et dernier points portent
    bien les contraintes extremes du nuage. Le programme trace les deux
    droites pour permettre de le verifier visuellement.
    """
    x, y, dx, dy = [np.asarray(v, dtype=float) for v in (x, y, dx, dy)]
    if x.ndim != 1 or not (x.shape == y.shape == dx.shape == dy.shape) or len(x) < 2:
        raise ValueError("Droites extremes : quatre vecteurs de meme taille sont attendus.")
    if np.any(dx < 0) or np.any(dy < 0):
        raise ValueError("Droites extremes : les incertitudes doivent etre positives.")

    # Premier et dernier points apres tri par R dans preparer_donnees().
    x1, y1, dx1, dy1 = x[0], y[0], dx[0], dy[0]
    x2, y2, dx2, dy2 = x[-1], y[-1], dx[-1], dy[-1]

    # Plus pentue : A bas-droite, B haut-gauche.
    Amax = (x1 + dx1, y1 - dy1)
    Bmax = (x2 - dx2, y2 + dy2)
    denom_max = Bmax[0] - Amax[0]
    if denom_max <= 0:
        raise ValueError("Droite extreme max : les intervalles horizontaux des extremites se recouvrent.")
    a_max_cand = (Bmax[1] - Amax[1]) / denom_max
    b_maxline = Amax[1] - a_max_cand * Amax[0]

    # Moins pentue : C haut-gauche, D bas-droite.
    Amin = (x1 - dx1, y1 + dy1)
    Bmin = (x2 + dx2, y2 - dy2)
    denom_min = Bmin[0] - Amin[0]
    if denom_min <= 0:
        raise ValueError("Droite extreme min : les intervalles horizontaux des extremites se recouvrent.")
    a_min_cand = (Bmin[1] - Amin[1]) / denom_min
    b_minline = Amin[1] - a_min_cand * Amin[0]

    # Dans le cas normal a_max_cand > a_min_cand. On garde quand meme une
    # sortie robuste si les donnees sont inhabituelles, sans perdre les paires
    # pente/interception correspondant aux deux droites effectivement tracees.
    lignes = sorted([(a_min_cand, b_minline, (Amin, Bmin)),
                     (a_max_cand, b_maxline, (Amax, Bmax))], key=lambda z: z[0])
    a_min, b_assoc_min, pts_min = lignes[0]
    a_max, b_assoc_max, pts_max = lignes[1]
    b_min = min(b_assoc_min, b_assoc_max)
    b_max = max(b_assoc_min, b_assoc_max)

    return DroitesExtremes(
        a_min=float(a_min), b_min=float(b_min),
        a_max=float(a_max), b_max=float(b_max),
        da=float((a_max - a_min) / 2),
        db=float((b_max - b_min) / 2),
        points_min=pts_min, points_max=pts_max,
    )


def valeur_incertitude(valeur: float, incertitude: float, unite: str) -> str:
    """Deux chiffres significatifs pour l'incertitude, meme rang pour la valeur."""
    if not np.isfinite(incertitude) or incertitude <= 0:
        return f"{valeur:.6g} +/- {incertitude:.3g} {unite}"
    decimales = 1 - int(np.floor(np.log10(incertitude)))
    v, d = round(float(valeur), decimales), round(float(incertitude), decimales)
    p = max(0, decimales)
    return f"({v:.{p}f} +/- {d:.{p}f}) {unite}"


# %% 3. FIGURES EXPERIMENTALES : UNE FIGURE PAR GRAPHIQUE

def nouvelle_figure(titre: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7.4, 4.8), layout="constrained")
    ax.set(title=titre, xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="out")
    return fig, ax


def enregistrer(fig, dossier: Path, nom: str, afficher: bool,
                fichiers: list[str], mention: str = "") -> None:
    if Path(nom).name != nom or not nom:
        raise ValueError("Le nom de figure ne doit pas contenir de chemin.")
    dossier.mkdir(parents=True, exist_ok=True)
    if mention:
        fig.suptitle(mention, fontsize=9, fontweight="bold")
    for extension in ("pdf", "png"):
        fichier = dossier / f"{nom}.{extension}"
        fig.savefig(fichier, dpi=200, bbox_inches="tight")
        fichiers.append(str(fichier.resolve()))
    if not afficher:
        plt.close(fig)


def ajouter_reference(ax, reference: tuple, symbole: str) -> None:
    valeur, inc = reference
    if valeur is None:
        return
    if not np.isfinite(valeur) or valeur <= 0:
        raise ValueError(f"Reference {symbole} : valeur positive attendue.")
    if inc is not None and (not np.isfinite(inc) or inc < 0):
        raise ValueError(f"Reference {symbole} : incertitude invalide.")
    ax.axhline(valeur, linestyle="-.", label=f"{symbole} constructeur")
    if inc is not None:
        ax.axhspan(valeur - inc, valeur + inc, alpha=0.10,
                   label="Intervalle constructeur")


def traiter_circuit(donnees: dict, circuit: str, dossier: Path, afficher: bool,
                    exemple: bool, journal: list[str], fichiers: list[str]) -> dict | None:
    serie = preparer_donnees(donnees, circuit, journal)
    if serie is None:
        return None
    R, dR, tau, dtau = serie
    rc = circuit == "RC"
    y, dy = (tau, dtau) if rc else (1 / tau, dtau / tau**2)
    fit = regression_additive(R, y, dR, dy)
    extremes = droites_extremes(R, y, dR, dy)
    if np.max(dy) / np.min(dy) > 5:
        avertir(f"{circuit} : incertitudes verticales tres inegales ; justifier l'OLS non pondere.", journal)
    if np.max(dR) > 0.05 * np.ptp(R):
        avertir(f"{circuit} : Delta R non faible devant l'etendue ; la methode OLS est a reexaminer.", journal)
    repere = dy + abs(fit.a) * dR
    n_ecarts = int(np.sum(np.abs(fit.residus) > repere))
    if n_ecarts:
        avertir(f"{circuit} : {n_ecarts} residu(s) hors du repere Delta y+|a|Delta R. "
                "Examiner mesures et modele ; ce n'est pas un test statistique.", journal)

    prefixe = circuit.lower()
    mention = "EXEMPLE 2025 - données d'un autre groupe" if exemple else ""
    etiquette = "Données publiées (2025)" if exemple else "Mesures"
    facteur = 1e6 if rc else 1.0
    ylabel = r"Constante de temps $\tau$ ($\mu$s)" if rc else r"Inverse $1/\tau$ (s$^{-1}$)"
    titre = "RC : ajustement de $\\tau(R)$" if rc else "RL : ajustement de $1/\\tau(R)$"
    R_grille = np.linspace(R.min(), R.max(), 500)
    fig, ax = nouvelle_figure(titre, r"Résistance $R$ ($\Omega$)", ylabel)
    ax.errorbar(R, y * facteur, xerr=dR, yerr=dy * facteur,
                fmt="o", markersize=4, capsize=3, label=etiquette)
    eq = f"Régression : y = {fit.a * facteur:.5g} R {fit.b * facteur:+.5g}"
    ax.plot(R_grille, (fit.a * R_grille + fit.b) * facteur, linewidth=2.0, label=eq)

    # Deux droites extremes utilisees pour a_min et a_max.
    ax.plot(R_grille,
            (extremes.a_max * R_grille + (extremes.points_max[0][1] - extremes.a_max * extremes.points_max[0][0])) * facteur,
            linestyle="--", linewidth=1.4,
            label=f"Plus pentue : a_max = {extremes.a_max * facteur:.5g}")
    ax.plot(R_grille,
            (extremes.a_min * R_grille + (extremes.points_min[0][1] - extremes.a_min * extremes.points_min[0][0])) * facteur,
            linestyle=":", linewidth=1.8,
            label=f"Moins pentue : a_min = {extremes.a_min * facteur:.5g}")

    # Points A-B et C-D servant a construire les deux droites extremes.
    pmax = np.asarray(extremes.points_max)
    pmin = np.asarray(extremes.points_min)
    ax.plot(pmax[:, 0], pmax[:, 1] * facteur, "x", markersize=7,
            label="Extrémités utilisées pour a_max")
    ax.plot(pmin[:, 0], pmin[:, 1] * facteur, "+", markersize=8,
            label="Extrémités utilisées pour a_min")

    unite_a = 'us/ohm' if rc else 's^-1/ohm'
    unite_b = 'us' if rc else 's^-1'
    texte = (f"a = {fit.a * facteur:.6g} {unite_a}\n"
             f"a_min = {extremes.a_min * facteur:.6g} ; a_max = {extremes.a_max * facteur:.6g}\n"
             f"Delta a = {extremes.da * facteur:.3g} {unite_a}\n"
             f"b = {fit.b * facteur:.6g} {unite_b} ; Delta b = {extremes.db * facteur:.3g} {unite_b}\n"
             f"R² = {fit.r2:.6f} ; {len(R)} points")
    ax.text(0.03, 0.97, texte, transform=ax.transAxes, va="top", fontsize=8.2,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.80, edgecolor="0.75"))
    ax.legend(loc="best", fontsize=7.4)
    enregistrer(fig, dossier, f"{prefixe}_regression", afficher, fichiers, mention)

    resultat = {"n": len(R), "a_SI": fit.a, "b_SI": fit.b,
                "a_min_SI": extremes.a_min, "a_max_SI": extremes.a_max,
                "b_min_SI": extremes.b_min, "b_max_SI": extremes.b_max,
                "Delta_a_SI": extremes.da, "Delta_b_SI": extremes.db,
                "Delta_a_additive_controle_SI": fit.da,
                "Delta_b_additive_controle_SI": fit.db,
                "R2": fit.r2 if np.isfinite(fit.r2) else None,
                "residus_SI": fit.residus.tolist(),
                "unite_a": "F" if rc else "H^-1", "unite_b": "s" if rc else "s^-1"}
    journal.extend([f"\n{circuit} - {len(R)} mesures", texte,
                    "Incertitudes finales sur a et b : methode des droites extremes.",
                    f"Controle propagation additive : Delta a = {fit.da * facteur:.6g}, Delta b = {fit.db * facteur:.6g}."])
    estimation = None
    if fit.a <= 0:
        avertir(f"{circuit} : pente non positive ; parametres physiques non deduits. "
                "La droite reste tracee pour discuter ce resultat.", journal)
    else:
        grandeur = fit.a if rc else 1 / fit.a
        dgrandeur = extremes.da if rc else extremes.da / fit.a**2
        Rp = fit.b / fit.a
        dRp = extremes.db / abs(fit.a) + abs(fit.b) * extremes.da / fit.a**2
        estimation = (grandeur, dgrandeur)
        resultat.update({"C_F" if rc else "L_H": grandeur,
                         "Delta_C_F" if rc else "Delta_L_H": dgrandeur,
                         "Rprime_ohm": Rp, "Delta_Rprime_ohm": dRp})
        echelle, unite = (1e6, "uF") if rc else (1e3, "mH")
        journal.append(("C = " if rc else "L = ") +
                       valeur_incertitude(grandeur * echelle, dgrandeur * echelle, unite))
        journal.append("R' = " + valeur_incertitude(Rp, dRp, "ohm"))
        if Rp < 0:
            avertir(f"{circuit} : R' centrale negative ; ne pas remplacer par sa valeur absolue.", journal)
        if extremes.da >= abs(fit.a):
            avertir(f"{circuit} : l'incertitude par droites extremes atteint la valeur de la pente ; "
                    "les quotients sont mal determines.", journal)

    # Calculs directs : toujours en SI, sans arrondis intermediaires.
    masque = R > 0
    Rpos, dRpos, tpos, dtpos = R[masque], dR[masque], tau[masque], dtau[masque]
    apparent = tpos / Rpos if rc else Rpos * tpos
    dapparent = dtpos / Rpos + tpos * dRpos / Rpos**2 if rc else Rpos * dtpos + tpos * dRpos
    echelle = 1e6 if rc else 1e3
    tableau = np.column_stack((Rpos, dRpos, tpos * 1e6, dtpos * 1e6,
                               apparent * echelle, dapparent * echelle))
    entete = "R_ohm;Delta_R_ohm;tau_us;Delta_tau_us;" + (
        "C_app_uF;Delta_C_app_uF" if rc else "L_app_mH;Delta_L_app_mH")
    ftable = dossier / f"{prefixe}_valeurs_calculees.csv"
    np.savetxt(ftable, tableau, delimiter=";", header=entete, comments="", fmt="%.10g")
    fichiers.append(str(ftable.resolve()))
    if not np.all(masque):
        journal.append(f"{circuit} : R=0 conserve en regression, exclu de l'estimation directe.")

    if TRACER_COMPLEMENTS:
        fig, ax = nouvelle_figure(f"{circuit} : résidus de l'ajustement", r"Résistance $R$ ($\Omega$)",
                                 r"Résidu ($\mu$s)" if rc else "Résidu (s$^{-1}$)")
        ax.errorbar(R, fit.residus * facteur, xerr=dR, yerr=repere * facteur,
                    fmt="o", capsize=3, markersize=4,
                    label=r"Barres : $\Delta y+|a|\Delta R$ (repère visuel)")
        ax.axhline(0, linestyle="--", label="Résidu nul")
        ax.legend(fontsize=8)
        enregistrer(fig, dossier, f"{prefixe}_residus", afficher, fichiers, mention)

        symbole = "C" if rc else "L"
        ylabel_app = r"Capacité apparente $C_{app}$ ($\mu$F)" if rc else r"Inductance apparente $L_{app}$ (mH)"
        fig, ax = nouvelle_figure(f"{circuit} : estimation directe sans $R'$", r"Résistance $R$ ($\Omega$)", ylabel_app)
        ax.errorbar(Rpos, apparent * echelle, xerr=dRpos, yerr=dapparent * echelle,
                    fmt="o", markersize=4, capsize=3, label=r"$C_{app}=\tau/R$" if rc else r"$L_{app}=R\tau$")
        if estimation:
            g, dg = estimation
            ax.axhline(g * echelle, linestyle="--", label=f"{symbole} issu de la régression")
            ax.axhspan((g - dg) * echelle, (g + dg) * echelle, alpha=0.12,
                       label="Intervalle propagé de la régression")
        ajouter_reference(ax, C_REFERENCE_UF if rc else L_REFERENCE_MH, symbole)
        ax.legend(fontsize=8)
        enregistrer(fig, dossier, f"{prefixe}_{symbole}_apparent", afficher, fichiers, mention)

        if not rc:
            fig, ax = nouvelle_figure(r"RL : dépendance non affine de $\tau(R)$",
                                     r"Résistance $R$ ($\Omega$)", r"Constante de temps $\tau$ ($\mu$s)")
            ax.errorbar(R, tau * 1e6, xerr=dR, yerr=dtau * 1e6,
                        fmt="o", capsize=3, markersize=4, label=etiquette)
            denom = fit.a * R_grille + fit.b
            modele = np.full_like(denom, np.nan)
            np.divide(1e6, denom, out=modele, where=denom > 0)
            if fit.a > 0:
                ax.plot(R_grille, modele, label=r"$\tau=1/(aR+b)$, issu de la régression")
            ax.legend(fontsize=8)
            enregistrer(fig, dossier, "rl_tau_R", afficher, fichiers, mention)
    return resultat


# %% 4. ILLUSTRATIONS THEORIQUES : AUCUNE ACQUISITION RECONSTITUEE

def signal_periodique(x: np.ndarray, forme: str) -> np.ndarray:
    """Signal de periode 1, centre, d'amplitude 1. x = t/T."""
    phase = np.mod(x, 1.0)
    if forme == "creneau":
        return np.where(phase < 0.5, 1.0, -1.0)
    if forme == "triangle":
        return 1.0 - 4.0 * np.abs(phase - 0.5)
    if forme == "sinus":
        return np.sin(2 * np.pi * phase)
    raise ValueError("Forme inconnue : creneau, triangle ou sinus.")


def reponse_periodique(forme: str, theta: float, periodes: int = 3, n: int = 4000):
    """Resolution de theta*z' + z = e en regime periodique etabli.

    Le signal est maintenu constant pendant chaque petit pas dx=1/n.
    z[k+1] = q*z[k] + (1-q)*e[k], q=exp(-dx/theta).
    Pour les signaux lisses, il s'agit d'une approximation d'echantillonnage.
    L'etat initial periodique est calcule sans transitoire de demarrage.
    RC : sortie = z. RL ideal : sortie = e-z.
    """
    if not np.isfinite(theta) or theta <= 0 or n < 100 or periodes < 1:
        raise ValueError("theta > 0, n >= 100 et periodes >= 1 sont necessaires.")
    # Au moins 100 echantillons par tau pour les constantes courtes.
    n = max(n, int(np.ceil(100 / theta)))
    if n > 200000:
        raise ValueError("tau/T trop petit pour cette illustration ; augmenter ce rapport.")
    dx = 1 / n
    alpha = -np.expm1(-dx / theta)
    q = 1 - alpha
    e_periode = signal_periodique(np.arange(n) / n, forme)
    z_fin = 0.0
    for e in e_periode:
        z_fin = q * z_fin + alpha * e
    z0 = z_fin / (-np.expm1(-1 / theta))
    x = np.arange(periodes * n + 1) / n
    entree = signal_periodique(x, forme)
    z = np.empty_like(x)
    z[0] = z0
    for k in range(len(x) - 1):
        z[k + 1] = q * z[k] + alpha * entree[k]
    return x, entree, z, n


def tracer_theorie(dossier: Path, afficher: bool, fichiers: list[str]) -> None:
    # Reponses a un echelon : temps normalise par tau, tensions par E.
    t = np.r_[np.linspace(-0.5, 0, 100, endpoint=False), np.linspace(0, 6, 1400)]
    e = (t >= 0).astype(float)
    for circuit in ("RC", "RL"):
        rc = circuit == "RC"
        u = np.zeros_like(t)
        u[t >= 0] = -np.expm1(-t[t >= 0]) if rc else np.exp(-t[t >= 0])
        seuil = 1 - np.exp(-1) if rc else np.exp(-1)
        fig, ax = nouvelle_figure(f"{circuit} : réponse théorique à un échelon",
                                 r"Temps $t/\tau$ (sans unité)", "Tension / E (sans unité)")
        ax.step(t, e, where="post", linestyle="--", label=r"Entrée $e/E$")
        ax.plot(t, u, label=r"$u_C/E$" if rc else r"$u_L/E$ (bobine idéale)")
        ax.axvline(1, linestyle=":", linewidth=1)
        ax.axhline(seuil, linestyle=":", linewidth=1)
        ax.plot([1], [seuil], "o", label=f"t = tau : {100 * seuil:.1f} %")
        ax.legend(fontsize=9)
        enregistrer(fig, dossier, f"{circuit.lower()}_echelon_theorique", afficher, fichiers,
                    "MODÈLE THÉORIQUE - pas une mesure")

    # Deux axes explicites : l'attenuation n'est pas effacee par normalisation.
    for circuit, theta in (("RC", TAU_SUR_T_INTEGRATEUR), ("RL", TAU_SUR_T_DERIVATEUR)):
        rc = circuit == "RC"
        if (rc and theta < 5) or (not rc and theta > 0.05):
            print("ATTENTION : les rapports tau/T choisis sont peu proches du regime limite.")
        for forme in ("creneau", "triangle", "sinus"):
            x, e, z, n = reponse_periodique(forme, theta)
            sortie = z if rc else e - z
            fonction = "intégrateur" if rc else "dérivateur idéal"
            nom_forme = "créneau" if forme == "creneau" else forme
            fig, ax = nouvelle_figure(f"{circuit} {fonction} - entrée {nom_forme}, $\\tau/T={theta:g}$",
                                     r"Temps $t/T$ (sans unité)", "Entrée e/E (axe gauche)")
            droite = ax.twinx()
            droite.set_ylabel(("Sortie $u_C/E$" if rc else "Sortie $u_L/E$") + " (axe droit)")
            ax.plot(x, e, linestyle="--", linewidth=1.1, label="Entrée centrée (gauche)")
            droite.plot(x, sortie, linewidth=1.4, label="Réponse du modèle (droite)")
            if rc:
                primitive = np.r_[0.0, np.cumsum(e[:-1]) / n] / theta
                primitive -= np.mean(primitive[:n])
                droite.plot(x, primitive, linestyle=":", linewidth=1.3,
                            label=r"Approximation $\tau^{-1}\int e\,dt$ (droite)")
            elif forme != "creneau":
                phase = np.mod(x, 1.0)
                derivee = 2 * np.pi * np.cos(2 * np.pi * x) if forme == "sinus" else np.where(phase < 0.5, 4.0, -4.0)
                if forme == "triangle":
                    derivee[np.isclose(phase, 0) | np.isclose(phase, 0.5)] = np.nan
                droite.plot(x, theta * derivee, linestyle=":", linewidth=1.3,
                            label=r"Approximation $\tau\,de/dt$ (droite)")
            else:
                ax.text(0.02, 0.03, "Créneau discontinu : la dérivée n'est pas une courbe ordinaire.",
                        transform=ax.transAxes, fontsize=8)
            ax.set_xlim(0, 3)
            ax.set_ylim(-1.35, 1.35)
            lignes1, labels1 = ax.get_legend_handles_labels()
            lignes2, labels2 = droite.get_legend_handles_labels()
            ax.legend(lignes1 + lignes2, labels1 + labels2, loc="upper center",
                      bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=8)
            nom = f"{circuit.lower()}_{'integrateur' if rc else 'derivateur'}_{forme}_theorie"
            enregistrer(fig, dossier, nom, afficher, fichiers,
                        "MODÈLE - régime périodique ; échelles verticales distinctes")


# %% 5. TRACES DE VOS ACQUISITIONS CSV (FORMAT EXPLICITE)

def lire_acquisition(config: dict):
    """Lire les colonnes choisies ; virgule decimale acceptee avec ';'.

    Pas de detection silencieuse du format constructeur : configurer les
    lignes d'entete, le separateur, les colonnes et l'unite temporelle.
    """
    fichier = Path(config["fichier"])
    if not fichier.is_absolute():
        fichier = RACINE / fichier
    colonnes = tuple(config.get("colonnes", (0, 1, 2)))
    if len(colonnes) != 3 or any(not isinstance(c, int) or c < 0 for c in colonnes):
        raise ValueError("Acquisition : trois numeros de colonnes >= 0 sont attendus.")
    valeurs = []
    with fichier.open("r", encoding="utf-8-sig", newline="") as f:
        lecteur = csv.reader(f, delimiter=config.get("separateur", ";"))
        for _ in range(config.get("lignes_entete", 1)):
            next(lecteur, None)
        for numero, ligne in enumerate(lecteur, start=config.get("lignes_entete", 1) + 1):
            if not ligne or all(not c.strip() for c in ligne):
                continue
            try:
                valeurs.append([float(ligne[c].strip().replace(",", ".")) for c in colonnes])
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{fichier.name}, ligne {numero} : verifier colonnes et entete.") from exc
    if len(valeurs) < 2:
        raise ValueError(f"{fichier.name} : au moins deux echantillons sont necessaires.")
    v = np.asarray(valeurs, dtype=float)
    facteur = {"s": 1.0, "ms": 1e-3, "us": 1e-6}
    unite = config.get("unite_temps", "s")
    if unite not in facteur:
        raise ValueError("unite_temps doit valoir 's', 'ms' ou 'us'.")
    t, ch1, ch2 = v.T.copy()
    t *= facteur[unite]
    if not np.isfinite(v).all() or np.any(np.diff(t) <= 0):
        raise ValueError(f"{fichier.name} : valeurs non finies ou temps non strictement croissant.")
    return t, ch1, ch2


def tracer_acquisition(config: dict, dossier: Path, afficher: bool, fichiers: list[str]) -> None:
    t, ch1, ch2 = lire_acquisition(config)
    circuit = config.get("circuit", "RC").upper()
    if circuit not in ("RC", "RL"):
        raise ValueError("Le circuit doit etre RC ou RL.")
    fig, ax = nouvelle_figure(config.get("titre", f"{circuit} : acquisition oscilloscope"),
                             "Temps t (ms)", "Tension (V)")
    ax_sortie = ax.twinx() if config.get("double_axe", False) else ax
    if ax_sortie is not ax:
        ax.set_ylabel("CH1 (V, axe gauche)")
        ax_sortie.set_ylabel("CH2 (V, axe droit)")
    ax.plot(t * 1e3, ch1, linestyle="--", linewidth=1.1, label="CH1 : entrée mesurée")
    ax_sortie.plot(t * 1e3, ch2, linewidth=1.1,
                   label="CH2 : condensateur" if circuit == "RC" else "CH2 : bobine réelle")
    p = config.get("pointage", {})
    cles = ("t0_s", "tau_s", "u_initial_V", "u_final_V")
    if p and all(p.get(k) is not None for k in cles):
        t0, tau, ui, uf = [float(p[k]) for k in cles]
        if not np.isfinite([t0, tau, ui, uf]).all() or tau <= 0 or ui == uf:
            raise ValueError("Pointage : tau > 0 et niveaux initial/final distincts et finis.")
        if not (t.min() <= t0 < t0 + tau <= t.max()):
            raise ValueError("Le pointage est hors de la plage temporelle de l'acquisition.")
        # Meme formule pour une montee ou une descente, plateau reel compris.
        u_tau = uf + (ui - uf) * np.exp(-1)
        ax.axvline(t0 * 1e3, linestyle=":", linewidth=1)
        ax.axvline((t0 + tau) * 1e3, linestyle=":", linewidth=1)
        ax_sortie.axhline(u_tau, linestyle=":", linewidth=1)
        ax_sortie.plot([(t0 + tau) * 1e3], [u_tau], "o", label="Seuil de pointage à t0 + tau")
    elif p and any(p.get(k) is not None for k in cles):
        raise ValueError("Pointage incomplet : renseigner les quatre valeurs ou les laisser a None.")
    h1, l1 = ax.get_legend_handles_labels()
    if ax_sortie is not ax:
        h2, l2 = ax_sortie.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8)
    else:
        ax.legend(fontsize=8)
    mention = "ACQUISITION - échelles verticales distinctes" if ax_sortie is not ax else "ACQUISITION EXPÉRIMENTALE"
    enregistrer(fig, dossier, config.get("nom", f"{circuit.lower()}_acquisition"), afficher, fichiers, mention)


# %% 6. EXECUTION

def main(exemple_2025: bool | None = None, afficher: bool | None = None,
         dossier_sortie: str | Path | None = None, theorie: bool | None = None) -> dict:
    exemple = EXEMPLE_2025 if exemple_2025 is None else exemple_2025
    afficher = AFFICHER if afficher is None else afficher
    theorie = TRACER_THEORIE if theorie is None else theorie
    base = Path(dossier_sortie) if dossier_sortie is not None else DOSSIER_SORTIE
    if exemple:
        base = base / "EXEMPLE_2025"
    base.mkdir(parents=True, exist_ok=True)
    journal = ["TP1 - exploitation des donnees",
               "MODE : EXEMPLE 2025, donnees d'un autre groupe" if exemple else "MODE : MESURES PERSONNELLES",
               f"NumPy {np.__version__} ; Matplotlib {matplotlib.__version__}",
               "Calculs en SI ; tau saisi en us, conversion automatique en secondes.",
               "Droite centrale : OLS non pondere ; incertitudes finales : droites extremes."]
    if exemple:
        journal.append("Source : TP_Laura.pdf, Bouchard-Mourier / Torres, pages 13 et 30.")
    print("\n".join(journal))
    fichiers, erreurs = [], []
    resultats = {"mode": "exemple_2025" if exemple else "mesures", "circuits": {}}
    rc, rl = donnees_2025() if exemple else (RC, RL)
    dossier_mesures = base / ("donnees_2025" if exemple else "mesures")
    dossier_mesures.mkdir(parents=True, exist_ok=True)
    for circuit, donnees in (("RC", rc), ("RL", rl)):
        try:
            resultats["circuits"][circuit] = traiter_circuit(
                donnees, circuit, dossier_mesures, afficher, exemple, journal, fichiers)
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
            erreurs.append(f"{circuit} : {exc}")
            avertir(erreurs[-1], journal)
    if theorie:
        tracer_theorie(base / "theorie", afficher, fichiers)
    # Les vraies acquisitions ne sont pas melangees au test des donnees 2025.
    if not exemple:
        for config in ACQUISITIONS:
            try:
                tracer_acquisition(config, base / "acquisitions", afficher, fichiers)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                erreurs.append(f"Acquisition : {exc}")
                avertir(erreurs[-1], journal)
    resultats["erreurs"] = erreurs
    resultats["fichiers_generes_cette_execution"] = fichiers
    bilan = base / "bilan_numerique.txt"
    bilan.write_text("\n".join(journal) + "\n\nFichiers de cette execution :\n" +
                     "\n".join(fichiers) + "\n", encoding="utf-8")
    (base / "resultats.json").write_text(json.dumps(resultats, ensure_ascii=False, indent=2,
                                                    allow_nan=False), encoding="utf-8")
    print("\n" + "\n".join(l for l in journal if l.startswith(("C =", "L =", "R' ="))))
    print(f"\nFigures et bilan : {base.resolve()}")
    print(f"{sum(f.endswith('.pdf') for f in fichiers)} figures, chacune en PDF et PNG.")
    if erreurs:
        print("Certaines sections n'ont pas abouti : consulter les messages et le bilan.")
    if afficher:
        plt.show()
    return resultats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exemple-2025", action="store_true", help="Utiliser les mesures publiees de 2025.")
    parser.add_argument("--sans-affichage", action="store_true", help="Enregistrer sans ouvrir les figures.")
    parser.add_argument("--sans-theorie", action="store_true", help="Ne pas generer les illustrations theoriques.")
    parser.add_argument("--sortie", type=Path, default=None, help="Dossier des resultats.")
    args = parser.parse_args()
    main(exemple_2025=args.exemple_2025 or EXEMPLE_2025,
         afficher=AFFICHER and not args.sans_affichage,
         dossier_sortie=args.sortie,
         theorie=TRACER_THEORIE and not args.sans_theorie)
