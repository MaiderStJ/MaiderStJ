import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import math
from copy import deepcopy

# Début du chronomètre
start_time = time.time()

# Paramètres du modèle
sectors = ["Agriculture", "Energy", "Housing", "Transport", "Industry", "Technology"]
num_firms = 120
num_banks = 10
num_households = 600
num_centralbank = 1
num_periods = 25
num_simulations = 3

# Initialisation structurelle des 600 ménages
random.seed(42)
np.random.seed(42)

# Avant les runs
GDP = []  

# =========================================================
# === PATCH 1: Caches pour les agrégats de consommation ===
# =========================================================

# Caches pour stocker les valeurs de C et G renvoyées par compute_firm_revenues_and_profits
C_households_value_cache = []
G_socialized_cache = []
scenarios_to_plot = []
# === Trace dette publique (comme gdp_records) ===
public_debt_records = []
base_wage_list = [0.2]  # salaire de base à t=0



# État persistant des politiques (peut être un dict global)
POLICY_STATE = {
    "brown_credit_initial_cap": None  # sera fixé au 1er t actif en post_growth
}

def _consecutive_active(active_list, t):
    k, i = 0, t
    while i >= 0 and active_list[i] == 1:
        k += 1
        i -= 1
    return k


def sigma_share_at_t(t, sid, PostGrowthActive, pg_cons):
    if PostGrowthActive is None or len(PostGrowthActive) <= t or PostGrowthActive[t] != 1:
        return 0.0
    target = float(pg_cons.get("sigma_target", {}).get(sid, 0.0))
    ramp   = max(1, int(pg_cons.get("ramp_periods", 3)))
    phase  = min(1.0, max(0.0, t / ramp))
    return target * phase

def prod_intensity_factor(t, sid, kind="energy"):
    pgc = globals().get("pg_cons", {}) or {}
    PGA = globals().get("PostGrowthActive")
    sigma_t = sigma_share_at_t(t, sid, PGA, pgc)
    if kind == "energy":
        mu = float(pgc.get("mu_energy", {}).get(sid, 1.0))
    else:
        mu = float(pgc.get("mu_materials", {}).get(sid, 1.0))
    # moyenne pondérée: (1-σ)*1 + σ*mu = 1 - σ*(1-mu)
    return 1.0 - sigma_t * (1.0 - mu)

def get_gdp_series():
    """Retourne la série du PIB, depuis GDP ou gdp_records."""
    if 'GDP' in globals() and isinstance(GDP, list):
        return GDP
    if 'gdp_records' in globals():
        return [ rec.get('GDP', 0.0) for rec in gdp_records ]
    return []


# Petit helper pour garantir la taille des listes
def _ensure_len(lst, n):
    if len(lst) < n:
        lst.extend([None] * (n - len(lst)))


# ===========================
# PARAMÈTRES & ÉTATS PUBLIC/BC
# ===========================

# --- Paramètres (ajuste à volonté) ---
PUBLIC_PARAMS = {
    # Fiscalité simple (peut devenir progressive plus tard)
    "tau_h": 0.10,     # taux moyen d'impôt sur revenu des ménages
    "tau_f": 0.15,     # taux moyen d'impôt sur profits des firmes (>=0)
    # Taux & cibles
    "pi_star": 0.02,   # cible d'inflation (2%)
    "phi_pi": 1.50,    # réaction du taux aux écarts d'inflation
    "phi_y": 0.10,     # réaction à l'écart d'activité (approximée)
    # Part de monétisation du déficit par la Banque Centrale
    # (le reste est financé par marchés/banques/ménages)
    "share_cb_monet": 0.40,
    # Taux sur obligations publiques & réserves (initialisation)
    "i_gov0": 0.02,
    "i_res0": 0.02,
    "cb_rate0": 0.02
}

# --- États Gouvernement / Banque centrale / Banques (listes par période) ---
Government = {
    "TaxesHouseholds": [],
    "TaxesFirms": [],
    "CarbonTaxRevenue": [],      # <-- Recettes de taxe carbone (agrégées)
    "G_current": [],             # Dépenses courantes (UB, emploi garanti, services sociaux, etc.)
    "G_capex": [],               # Dépenses d'invest public (dont I_pub_shared)
    "PrimaryDeficit": [],
    "InterestGov": [],
    "Deficit": [],
    "PublicDebt": [],            # Stock de dette publique
    "i_gov": []                  # Taux moyen sur dette publique
}

CentralBank = {
    "cb_rate": [],
    "i_res": [],
    "GovBonds_CB": [],          # Titres publics détenus par la BC (QE / monétisation)
    "BankReserves": [],         # Réserves des banques à la BC (passif BC)
    "GovAccount": []            # Compte du Trésor à la BC (passif BC)
}

BanksAggregate = {
    "GovBonds_Banks": [],       # Portefeuille de titres publics (côté banques)
    "Deposits_HH": [],          # Dépôts ménages agrégés (si tu veux les suivre ici)
    "Deposits_Firms": [],       # Dépôts firmes agrégés (optionnel)
    "Reserves": [],             # Réserves détenues à la BC (même grandeur que CentralBank["BankReserves"])
    "Profits_InterestOnReserves": []
}

# ---------- Helpers sécurisés ----------
def _val(x, default=0.0):
    """Retourne x si non-None, sinon default."""
    return default if x is None else x

def _pos(x):
    """Partie positive."""
    return x if x > 0 else 0.0

def get_banks_profits_to_gdp_ratio(t):
    """
    Retourne BanksProfits/GDP à la période t.
    Ici, on prend comme proxy les profits des banques provenant des
    intérêts sur réserves : BanksAggregate["Profits_InterestOnReserves"].
    """
    # Série du PIB (via GDP ou gdp_records)
    try:
        gdp_series = get_gdp_series()
    except Exception:
        gdp_series = []

    if t < 0 or t >= len(gdp_series):
        return 0.0

    gdp_t = gdp_series[t] or 0.0
    if gdp_t <= 0:
        return 0.0

    # Profits des banques (proxy) : Profits_InterestOnReserves
    profits_list = []
    if "BanksAggregate" in globals():
        profits_list = BanksAggregate.get("Profits_InterestOnReserves", [])

    if not profits_list or len(profits_list) <= t:
        return 0.0

    profits_t = profits_list[t] or 0.0
    return profits_t / gdp_t

# ===========================
# RÈGLE DE TAUX (simple & robuste)
# ===========================
def policy_central_bank_rate(t, inflation_t, gdp_t, gdp_trend_t):
    """
    Règle de type Taylor: i_t = i_{t-1} + φπ(π-π*) + φy(output_gap)
    - inflation_t : inflation observée (ex: general_inflation[t])
    - gdp_t : PIB courant
    - gdp_trend_t : tendance/PIB potentiel (met  gdp_t si tu n'as pas de tendance)
    """
    pi_star = PUBLIC_PARAMS["pi_star"]
    phi_pi  = PUBLIC_PARAMS["phi_pi"]
    phi_y   = PUBLIC_PARAMS["phi_y"]

    gap_y = 0.0
    if gdp_trend_t is not None and gdp_trend_t != 0:
        gap_y = (gdp_t - gdp_trend_t) / gdp_trend_t

    prev_cb = CentralBank["cb_rate"][t-1] if t > 0 and len(CentralBank["cb_rate"]) > 0 else PUBLIC_PARAMS["cb_rate0"]
    new_cb  = prev_cb + phi_pi * (_val(inflation_t) - pi_star) + phi_y * gap_y

    # IOER (intérêts sur réserves) : on fixe un plancher ~ taux directeur
    i_res = new_cb  # tu peux mettre une marge si tu veux un corridor
    return max(new_cb, 0.0), max(i_res, 0.0)

# ===========================
# CYCLE PUBLIC/BC (flux -> stocks)
# ===========================
def update_government_and_cb_accounts(
    t,
    # --- Entrées agrégées venant de TON code existant (ou 0 si absentes) ---
    total_public_spending_t,   # ex: total_public_spending[t]
    g_shared_t,                # ex: G_shared_list[t] (volet "services socialisés" OPEX)
    i_pub_shared_t,            # ex: I_pub_shared_list[t] (CAPEX public)
    household_income_agg_t,    # somme des revenus bruts ménages (avant impôts)
    firms_profits_agg_t,       # profits agrégés des firmes (>=0 sinon 0)
    carbon_tax_revenue_t,      # <-- TAUX CARBONE AGRÉGÉ COLLECTÉ AU T (NOUVEAU)
    inflation_t,               # ex: general_inflation[t]
    gdp_t,                     # ex: GDP agrégé de ta période t
    gdp_trend_t=None           # mets gdp_t si tu n'as pas encore de tendance
):
    """
    Étapes:
    1) Calcule les recettes publiques (impôts ménages, impôts firmes, taxe carbone)
    2) Ventile dépenses en G_current (OPEX) vs G_capex (CAPEX public)
    3) Solde primaire, intérêts sur dette, déficit, dette
    4) Monétisation par la BC: achat de titres publics → +Réserves bancaires & +Bonds_BC
    5) Règle de taux: met à jour cb_rate et i_res; crédite intérêts sur réserves (banques)
    6) Comptabilise intérêts payés par l'État sur dette (i_gov)
    """

    # --- 0) Taux initiaux (si t=0) ---
    prev_debt = Government["PublicDebt"][t-1] if t > 0 and len(Government["PublicDebt"]) > 0 else 0.0
    prev_i_gov = Government["i_gov"][t-1] if t > 0 and len(Government["i_gov"]) > 0 else PUBLIC_PARAMS["i_gov0"]

    # --- 1) Recettes publiques (incluant taxe carbone) ---
    tau_h = PUBLIC_PARAMS["tau_h"]
    tau_f = PUBLIC_PARAMS["tau_f"]

    taxes_h = tau_h * _pos(_val(household_income_agg_t))
    taxes_f = tau_f * _pos(_val(firms_profits_agg_t))
    carbon_rev = _pos(_val(carbon_tax_revenue_t))  # <-- Recette taxe carbone (déjà agrégée)

    Government["TaxesHouseholds"].append(taxes_h)
    Government["TaxesFirms"].append(taxes_f)
    Government["CarbonTaxRevenue"].append(carbon_rev)

    total_revenues = taxes_h + taxes_f + carbon_rev

    # --- 2) Dépenses publiques (ventilation simple OPEX/CAPEX) ---
    # Idée: total_public_spending_t contient déjà UB, emploi garanti, subventions, etc.
    # On range "i_pub_shared_t" comme CAPEX public; le reste en OPEX avec g_shared_t
    g_capex_t = _pos(_val(i_pub_shared_t))
    # OPEX: tout le reste
    g_current_t = _pos(_val(total_public_spending_t) - g_capex_t)

    Government["G_current"].append(g_current_t)
    Government["G_capex"].append(g_capex_t)

    # --- 3) Solde primaire & déficit total ---
    primary_deficit = (g_current_t + g_capex_t) - total_revenues
    interest_gov = prev_i_gov * prev_debt
    deficit_t = primary_deficit + interest_gov

    Government["PrimaryDeficit"].append(primary_deficit)
    Government["InterestGov"].append(interest_gov)
    Government["Deficit"].append(deficit_t)

    # --- 4) Règle de monétisation: part du déficit financée par la BC (QE/achat primaire/secondaire) ---
    share_cb = PUBLIC_PARAMS["share_cb_monet"]
    monetization_t = _pos(deficit_t) * min(max(share_cb, 0.0), 1.0)

    # Mise à jour dette publique (émission nette = déficit - monétisation par BC)
    new_debt = prev_debt + deficit_t - monetization_t
    Government["PublicDebt"].append(max(new_debt, 0.0))  # pas de dette négative
    # Mise à jour taux moyen sur dette publique (option simple: lier à cb_rate avec spread)
    # On met à jour après avoir calculé cb_rate ci-dessous.

    # --- 5) Politique monétaire: calcule cb_rate & i_res (règle Taylor) ---
    new_cb_rate, new_i_res = policy_central_bank_rate(t, inflation_t, _val(gdp_t), _val(gdp_trend_t, _val(gdp_t)))
    CentralBank["cb_rate"].append(new_cb_rate)
    CentralBank["i_res"].append(new_i_res)

    # --- 6) Bilan BC & Banques: enregistrement monétisation => +Bonds_BC ; +Reserves Bancaires
    prev_bonds_cb = CentralBank["GovBonds_CB"][t-1] if t > 0 and len(CentralBank["GovBonds_CB"]) > 0 else 0.0
    prev_reserves = CentralBank["BankReserves"][t-1] if t > 0 and len(CentralBank["BankReserves"]) > 0 else 0.0
    prev_gov_acc  = CentralBank["GovAccount"][t-1] if t > 0 and len(CentralBank["GovAccount"])  > 0 else 0.0

    bonds_cb_t = prev_bonds_cb + monetization_t
    reserves_t = prev_reserves + monetization_t  # contrepartie monétaire
    gov_acc_t  = prev_gov_acc  # si tu veux faire transiter les paiements par GovAccount, on peut le gérer plus finement

    CentralBank["GovBonds_CB"].append(bonds_cb_t)
    CentralBank["BankReserves"].append(reserves_t)
    CentralBank["GovAccount"].append(gov_acc_t)

    # Côté banques (agrégé): si la BC achète, les banques voient leurs réserves augmenter
    prev_bonds_banks = BanksAggregate["GovBonds_Banks"][t-1] if t > 0 and len(BanksAggregate["GovBonds_Banks"]) > 0 else 0.0
    BanksAggregate["GovBonds_Banks"].append(max(prev_bonds_banks - monetization_t, 0.0))  # hypothèse: achats sur marché secondaire
    BanksAggregate["Reserves"].append(reserves_t)

    # --- 7) Intérêts sur réserves payés aux banques (coût pour BC, revenu banque) ---
    prev_res_for_interest = CentralBank["BankReserves"][t-1] if t > 0 and len(CentralBank["BankReserves"]) > 0 else prev_reserves
    ior_income = new_i_res * prev_res_for_interest
    BanksAggregate["Profits_InterestOnReserves"].append(ior_income)
    # Note: comptablement, ce coût n'est pas une dépense budgétaire du gouvernement mais un transfert intra-secteur "monétaire".
    # Tu peux, si tu le souhaites, modéliser un versement net de la BC à l'État ou l'inverse via dividendes.

    # --- 8) Met à jour le taux payé par l'État (spread simple sur cb_rate) ---
    # Exemple: i_gov = cb_rate + 0.5% * (1 + ratio_dette/PIB) borné >= 0
    gdp_safe = _val(gdp_t)
    den = gdp_safe if gdp_safe > 0 else 1.0
    debt_ratio = (Government["PublicDebt"][t] / den) if gdp_safe > 0 else 0.0
    i_gov_t = max(new_cb_rate + 0.005 * (1.0 + debt_ratio), 0.0)

def compute_carbon_tax_revenue_t(t, firm_list):
    total = 0.0
    for f in firm_list:
        if "CarbonTaxPaid" in f and len(f["CarbonTaxPaid"]) > t:
            total += f["CarbonTaxPaid"][t] or 0.0
    return total


def _safe_len(seq):
    try:
        return len(seq)
    except Exception:
        return 0

def _as_list_or_none(x):
    """Renvoie x si c'est une séquence indexable, sinon None."""
    try:
        _ = len(x)
        _ = x[0] if len(x) > 0 else None
        return x
    except Exception:
        return None

# === Helpers robustes ===
def _safe_list(x, n=None, fill=0.0):
    """Retourne une liste; tronque/pad à n si demandé."""
    if isinstance(x, list):
        if n is None: 
            return x
        y = x[:n]
        if len(y) < n:
            y = y + [fill] * (n - len(y))
        return y
    return [x] * (n if n is not None else 1)

def _get_gdp_series_from_records(gdp_records, num_periods=None):
    gdp = [rec.get("GDP", 0.0) for rec in gdp_records] if isinstance(gdp_records, list) else []
    if num_periods is not None:
        gdp = _safe_list(gdp, num_periods)
    return gdp

def build_public_debt_df_from_scenarios(scenarios_to_plot, num_periods):
    """
    Construit un DataFrame long avec : Period, ScenarioName, PublicDebt, DebtToGDP_pct,
    i_gov_pct, GovBonds_CB, GovBonds_Banks.
    - Si Government['i_gov'] est vide, on le reconstruit :
      i_gov_t = cb_rate_t + 0.005 * (1 + PublicDebt_t / GDP_t)
      (même logique que ton update_government_and_cb_accounts)
    """
    rows = []
    T = num_periods + 1  # attention: beaucoup de séries vont de 0 à num_periods

    for sc in scenarios_to_plot:
        scen = sc.get("name", "unknown")
        G = sc.get("G", {})
        CB = sc.get("CB", {})
        B = sc.get("B", {})
        gdp_series = _safe_list(sc.get("GDP", []), T, 0.0)

        debt = _safe_list(G.get("PublicDebt", []), T, 0.0)
        igov = _safe_list(G.get("i_gov", []), T, None)   # peut être vide
        cb_rate = _safe_list(CB.get("cb_rate", []), T, 0.0)
        cb_hold = _safe_list(CB.get("GovBonds_CB", []), T, 0.0)
        bank_hold = _safe_list(B.get("GovBonds_Banks", []), T, 0.0)

        # Rebuild i_gov si manquant, avec la formule de ton code
        # i_gov_t = cb_rate_t + 0.005 * (1 + debt_ratio), où debt_ratio = debt / GDP
        rebuilt_igov = []
        for t in range(T):
            if igov[t] is None:
                gdp_t = gdp_series[t] if t < len(gdp_series) else 0.0
                den = gdp_t if gdp_t > 0 else 1.0
                debt_ratio = (debt[t] / den) if gdp_t > 0 else 0.0
                ig = max(cb_rate[t] + 0.005 * (1.0 + debt_ratio), 0.0)
                rebuilt_igov.append(ig)
            else:
                rebuilt_igov.append(igov[t])

        for t in range(T):
            gdp_t = gdp_series[t] if t < len(gdp_series) else 0.0
            d2g = 100.0 * (debt[t] / gdp_t) if gdp_t else 0.0
            rows.append({
                "ScenarioName": scen,
                "Period": t,
                "PublicDebt": debt[t],
                "DebtToGDP_pct": d2g,
                "i_gov_pct": 100.0 * rebuilt_igov[t],
                "GovBonds_CB": cb_hold[t],
                "GovBonds_Banks": bank_hold[t],
            })

    df = pd.DataFrame(rows)

    # (Option) on exclut souvent Period == 0 dans les plots
    return df


params = {
    # Coefficients de cycle du carbone
    "phi_11": 0.6,
    "phi_21": 0.25,
    "phi_12": 0.35,
    "phi_22": 0.45,
    "phi_32": 0.1,
    "phi_23": 0.15,
    "phi_33": 0.7,
    
    # Forçage radiatif
    "RF_2CO2": 3.7,
    
    # Concentration CO2 pré-industrielle (ppm)
    "AtmosphericPreIndustrialCO2Concentration": 280,
    
    # Sensibilité climatique (°C)
    "CS": 3.0,
    
    # Coefficients temps (discrétisation température)
    "t_1": 0.2,
    "t_2": 0.1,
    "t_3": 0.05,
    
    # Coefficients fonction dommages climatiques
    "eta_1": 0.01,
    "eta_2": 0.005,
    "eta_3": 0.0001,
    
    # Emissions industrielles : part énergie non renouvelable
    "emissions_share": 0.8,
    
    # Changement d'usage des sols (taux de déclin émissions terres)
    "land_use_change": 0.01,

    "epsilon_V": 0.1,  # Exemple valeur : intensité énergétique faible pour capital vert
    "epsilon_B": 0.3,  # Exemple valeur : intensité énergétique plus élevée pour capital brun

    "A": 0.5,
    "B": 1.0,
    "C": 1.0,
    "D": 1.0
}

# === [NOUVEAU] Paramètres de socialisation de la consommation (activés seulement en post_growth) ===
social = {
    # part socialisée cible par secteur (1: Agri, 2: Énergie, 3: Logement, 4: Transport, 5: Industrie, 6: Tech)
    # social (remplace ces valeurs si déjà posées)
    "target_share": {1: 0.6, 2: 1, 3: 0.75, 4: 0.70, 5: 0.25, 6: 0.25},
    "ramp_periods": 3,  # au lieu de 12
    "hysteresis": 0.10,         # bande de variation max d'une période à l'autre pour la part allouée aux firmes
    "admin_price_factor": 0.60, # prix administré relatif au prix privé moyen (stabilise sans être gratuit)
    "min_public_fill": 0.10,    # si capacité privée shared insuffisante, comblement public minimal
}

# Traces agrégées (PIB & politiques)
G_shared_list = [0.0 for _ in range(num_periods)]    # dépense publique de conso socialisée (opex)
I_pub_shared_list = [0.0 for _ in range(num_periods)]# capex public pour shared (placeholder, peut rester 0 au début)

# === Décommodification : valeur privée retirée de C quand on bascule au shared ===
C_shifted_private_value_list = [0.0 for _ in range(num_periods)]

# === Plafond budgétaire pour l’investissement public partagé (lissage) ===
ksh_budget = {
    "cap_per_period": 0.2,   # ex: ≤ 5% du PIB (t-1)
}

# Seau de capacité budgétaire par période (rempli au besoin)
caproom_left = [float("inf")] * num_periods


# === [NOUVEAU] Paramètres K_shared par SECTEUR (1..6) ===
# 1 Agri, 2 Énergie, 3 Logement, 4 Transport, 5 Industrie, 6 Tech/Com
ksh_sector_params = {
    "delta": {1:0.04, 2:0.05, 3:0.02, 4:0.06, 5:0.05, 6:0.10},        # dépréciation effective
    "phi_service": {1:1.0, 2:1.0, 3:1.0, 4:1.0, 5:1.0, 6:1.0},        # services/unités de K
    "capex_per_unit": {1:1.0, 2:1.2, 3:1.5, 4:1.3, 5:1.1, 6:0.9},     # coût par unité de K
    "build_lag": {1:1, 2:2, 3:2, 4:2, 5:1, 6:1},
}

# Part de "capital public partagé" initiale vs capital privé sectoriel à t=0 (semis)
# (Valable uniquement en scénario post_growth)
seed_ratio_by_sector = {1:0.05, 2:0.50, 3:0.30, 4:0.15, 5:0.03, 6:0.02}

# États K_shared par secteur (1..6)
K_shared = {s: [0.0] * num_periods for s in range(1, 7)}
I_pub_shared_by_sector = {s: [0.0] * num_periods for s in range(1, 7)}

# === Achat épisodique & marché de l'occasion (activé tous scénarios) ===
purchase_freq = {5: 8, 6: 4}     # n périodes entre 2 achats: Industrie=8, Tech=4 (si période≈1 an)
used_params = {
    "target_used_share": {5: 0.50, 6: 0.50},  # à maturité : 30% d'occasion en Industrie, 50% en Tech
    "ramp_periods": 3,                        # rampe de montée de l'occasion
    "used_price_factor": 0.50,                # prix payé pour l'occasion ≈ 50% du neuf
    "divert_to_repair_when_suppressed": 0.50  # part de la demande “supprimée” convertie en réparation
}




# === Variante "post_growth" de l'équation de consommation ===
pg_cons = {
    # part socialisée visée (rampe)
    "sigma_target": {1:0.30, 2:0.60, 3:0.60, 4:0.60, 5:0.50, 6:0.00},
    "ramp_periods": 3,

    # gain d'efficacité en unités (partage)
    "eta": {1:1.05, 2:1.2, 3:1.25, 4:1.40, 5:1.00, 6:1.00},

    # prix administrés (en % du prix de marché)
    "alpha_admin": {1:0.80, 2:0.55, 3:0.60, 4:0.50, 5:1.00, 6:1.00},

    # co-pay ménage (part du prix admin payée par le ménage)
    "copay": {1:0.30, 2:0.20, 3:0.25, 4:0.30, 5:1.00, 6:1.00}
}


# Sobriété/efficacité côté PRODUCTION quand la part socialisée ↑ (valeurs proposées)
# sid: 1 Agri, 2 Énergie (usage final côté branches), 3 Logement, 4 Transports, 5 Industrie (biens), 6 Tech/Services
pg_cons.update({
    "mu_energy":    {1: 0.80, 2: 0.75, 3: 0.70, 4: 0.40, 5: 0.80, 6: 0.95},  # ↓ intensité énergétique (meilleur = plus bas)
    "mu_materials": {1: 0.80, 2: 0.80, 3: 0.60, 4: 0.60, 5: 0.75, 6: 0.90},  # ↓ intensité matière
})


paramsinnov = {
    # Adoption / régulation
    "Alpha_Regu": 0.05,
    "Sunset_Date": 10,
    "Revision_Period": 5,

    # R&D
    "RDshare": 0.2,
    "scale": 0.001,

    # Coefficients d’innovation
    "Alpha_X": 0.05,
    "Alpha_Eff": 0.05,
    "Alpha_Tox": 0.05,
    "Alpha_Bio": 0.05,

    # Bornes produit 1
    "Xmax_Prod1": 2.0,
    "Effmax_Prod1": 2.0,
    "Toxmin_Prod1": 0.1,
    "Biomin_Prod1": 0.1,

    # Bornes produit 2
    "Xmax_Prod2": 3.0,
    "Effmax_Prod2": 3.0,
    "Toxmin_Prod2": 0.05,
    "Biomin_Prod2": 0.05,

    # Valeurs initiales P1
    "Product1_Init_X": 1.0,
    "Product1_Init_Eff": 1.0,
    "Product1_Init_Tox": 1.0,
    "Product1_Init_Bio": 1.0,

    # Valeurs initiales P2
    "Product2_Init_X": 0.5,
    "Product2_Init_Eff": 0.5,
    "Product2_Init_Tox": 0.8,
    "Product2_Init_Bio": 0.8,

    # Cibles régulation
    "Target_Eff": 2.0,
    "Target_X": 2.0,
}


# Initialisation des variables biophysiques sur toute la durée des périodes
# (à adapter au nombre total de périodes de ta simulation)
num_periods = 25  # ou récupère la valeur réelle de ton modèle

nature = {
    "AtmosphericCO2Concentration": [params["AtmosphericPreIndustrialCO2Concentration"]] * num_periods,
    "BiosphereCO2Concentration": [150] * num_periods,  # estimation initiale
    "LowerOceansCO2Concentration": [90] * num_periods,  # estimation initiale
    "IndustrialEmissions": [0.0] * num_periods,
    "LandUseEmissions": [0.0] * num_periods,
    "TotalEmissions": [0.0] * num_periods,
    "AtmosphericTemperature": [1.0] * num_periods,  # °C absolue ou anomaly selon ton modèle
    "LowerOceansTemperature": [0.0068] * num_periods,
    "RadiativeForcing": [0.0] * num_periods,
    "DamagesFunction": [0.0] * num_periods,
}


    # Part de consommation normalisée par secteur pour les territoires ruraux
rural_prefs = {
    "AgPref": 0.24,
    "EnerPref": 0.19,
    "HousPref": 0.19,
    "TransPref": 0.24,
    "IndPref": 0.09,
    "TCPref": 0.05
}

# Part de consommation normalisée par secteur pour les territoires urbains
urban_prefs = {
    "AgPref": 0.21,
    "EnerPref": 0.16,
    "HousPref": 0.29,
    "TransPref": 0.19,
    "IndPref": 0.1,
    "TCPref": 0.05
}

# Initialisation des 600 ménages avec statut, compétences et préférences
households_struct = []

# --- Calibration rural / urbain à partir des données UE27 ---
rural_employed = 116_014_000
urban_employed = 93_567_000
rural_share = rural_employed / (rural_employed + urban_employed)   # ≈ 0.554
urban_share = 1.0 - rural_share                                    # ≈ 0.446

# === EMPLOI NON QUALIFIÉ ===
# Statut 1 : emploi brun non qualifié
for _ in range(227):
    status, skill = 1, 0
    id_territory = np.random.choice([0, 1], p=[rural_share, urban_share])
    prefs = rural_prefs if id_territory == 0 else urban_prefs
    h = {"Status": status, "SkillStatus": skill, "IdTerritory": id_territory}
    h.update(prefs)
    households_struct.append(h)

# Statut 2 : emploi vert non qualifié
for _ in range(84):
    status, skill = 2, 0
    id_territory = np.random.choice([0, 1], p=[rural_share, urban_share])
    prefs = rural_prefs if id_territory == 0 else urban_prefs
    h = {"Status": status, "SkillStatus": skill, "IdTerritory": id_territory}
    h.update(prefs)
    households_struct.append(h)

# === CHÔMAGE ===
# Statut 3 : chômage (non qualifié)
for _ in range(29):
    status, skill = 3, 0
    id_territory = np.random.choice([0, 1], p=[rural_share, urban_share])
    prefs = rural_prefs if id_territory == 0 else urban_prefs
    h = {"Status": status, "SkillStatus": skill, "IdTerritory": id_territory}
    h.update(prefs)
    households_struct.append(h)

# Statut 3 : chômage (qualifié)
for _ in range(13):
    status, skill = 3, 1
    id_territory = np.random.choice([0, 1], p=[rural_share, urban_share])
    prefs = rural_prefs if id_territory == 0 else urban_prefs
    h = {"Status": status, "SkillStatus": skill, "IdTerritory": id_territory}
    h.update(prefs)
    households_struct.append(h)

# === EMPLOI QUALIFIÉ ===
# Statut 5 : emploi brun qualifié
for _ in range(122):
    status, skill = 5, 1
    id_territory = np.random.choice([0, 1], p=[rural_share, urban_share])
    prefs = rural_prefs if id_territory == 0 else urban_prefs
    h = {"Status": status, "SkillStatus": skill, "IdTerritory": id_territory}
    h.update(prefs)
    households_struct.append(h)

# Statut 6 : emploi vert qualifié
for _ in range(125):
    status, skill = 6, 1
    id_territory = np.random.choice([0, 1], p=[rural_share, urban_share])
    prefs = rural_prefs if id_territory == 0 else urban_prefs
    h = {"Status": status, "SkillStatus": skill, "IdTerritory": id_territory}
    h.update(prefs)
    households_struct.append(h)

# Mélange aléatoire pour éviter les biais
random.shuffle(households_struct)

# IdTerritory (0 = rural, 1 = urbain) – cohérent avec les parts UE27
for h in households_struct:
    h["IdTerritory"] = np.random.choice([0, 1], p=[rural_share, urban_share])
# IdEmployer : si emploi, alors affectation à une firme
for h in households_struct:
    if h["Status"] in [1, 2, 5, 6]:
        h["IdEmployer"] = random.randint(0, num_firms - 1)
    else:
        h["IdEmployer"] = -1

general_inflation = [0.02 for _ in range(num_periods)]
price_ag_list = [1.0]
price_ener_list = [1.0]
price_hous_list = [1.0]
price_trans_list = [1.0]
price_ind_list = [1.0]
price_tech_list = [1.0]
CarbonTaxActive = []
TransitionActive = []
PostGrowthActive = []
growth_v = [0.01]  # croissance verte initiale pour t = 0
brown_loan_cap = [1.0]  # autorisation initiale à 100%
gdp_records = []
needsindex_records = []
social_cost_records = []
status_share_records = []
policy_outcomes = []
policy_records = []
household_records = []
vote_records = []

# Budget libéré non redépensé (trace) et taux de neutralisation du rebond
no_respend_rate = 0.7   # 70 % du budget libéré n’est pas redépensé sur le marché
SocializationSavings_list = [0.0 for _ in range(num_periods)]

# --- Traces conso en valeur (avant/après socialisation) ---
C_pre_social_list   = [0.0 for _ in range(num_periods)]
C_market_value_list = [0.0 for _ in range(num_periods)]

# --- Décommodification / socialisation ---
C_shifted_private_value_list = [0.0 for _ in range(num_periods)]
G_shared_list      = [0.0 for _ in range(num_periods)]
I_pub_shared_list  = [0.0 for _ in range(num_periods)]
SocializationSavings_list = [0.0 for _ in range(num_periods)]

# --- Split neuf / réparation par secteur ---
New_value_by_sector    = {s: [0.0]*num_periods for s in range(1,7)}
Repair_value_by_sector = {s: [0.0]*num_periods for s in range(1,7)}

# --- Traces suffisance / split neuf-réparation (DOIT être avant la fonction) ---
New_value_by_sector    = {s: [0.0]*num_periods for s in range(1,7)}
Repair_value_by_sector = {s: [0.0]*num_periods for s in range(1,7)}

# --- Listes liées à la socialisation/décommodification (déjà ajoutées plus haut) ---
C_pre_social_list            = [0.0 for _ in range(num_periods)]
C_market_value_list          = [0.0 for _ in range(num_periods)]
C_shifted_private_value_list = [0.0 for _ in range(num_periods)]
G_shared_list                = [0.0 for _ in range(num_periods)]
I_pub_shared_list            = [0.0 for _ in range(num_periods)]
SocializationSavings_list    = [0.0 for _ in range(num_periods)]

is_carbon_tax_scenario = False

def plot_gdp_components(gdp_df):

    expected_cols = ["CTot", "ITot", "GTot"]
    available_cols = [c for c in expected_cols if c in gdp_df.columns]

    if "ITot" not in available_cols:
        print("⚠️ Attention : ITot manquant dans gdp_df au moment du plot !")

    # ⚡ Filtrer pour exclure la période 0
    gdp_df = gdp_df[gdp_df["Period"] > 0]

    df_comp = gdp_df.melt(
        id_vars=["ScenarioName", "Period"],
        value_vars=available_cols,
        var_name="Composante",
        value_name="Valeur"
    )

    # Renommer pour lisibilité
    rename_map = {"CTot": "Consommation", "ITot": "Investissement", "GTot": "Dépenses publiques"}
    df_comp["Composante"] = df_comp["Composante"].replace(rename_map)

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df_comp,
        x="Period",
        y="Valeur",
        hue="Composante",
        style="ScenarioName",
        linewidth=2
    )
    plt.title("Décomposition du PIB par scénario")
    plt.xlabel("Période")
    plt.ylabel("Valeur agrégée")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def compute_firm_energy_intensity(firm, t, epsilon_V, epsilon_B):
    green_cap = firm["GreenCapital"][t] if t < len(firm["GreenCapital"]) else 0
    brown_cap = firm["BrownCapital"][t] if t < len(firm["BrownCapital"]) else 0
    total_cap = green_cap + brown_cap
    green_ratio = green_cap / total_cap if total_cap > 0 else 0

    # intensité de base (ta logique)
    intensity_base = green_ratio * epsilon_V + (1 - green_ratio) * epsilon_B

    # ajustement PRODUCTION-BASED par secteur (post-growth socialisé)
    sid = int(firm.get("SectorId", 5))  # fallback industrie
    factor = prod_intensity_factor(t, sid, kind="energy")

    return intensity_base * factor

def update_biophys_vars(nature, firms, t, params):
    import math

    CO2_prev = nature["AtmosphericCO2Concentration"][t-1] if t > 0 else nature["AtmosphericCO2Concentration"][0]
    Biosphere_CO2_prev = nature["BiosphereCO2Concentration"][t-1] if t > 0 else nature["BiosphereCO2Concentration"][0]
    LowerOceans_CO2_prev = nature["LowerOceansCO2Concentration"][t-1] if t > 0 else nature["LowerOceansCO2Concentration"][0]
    Temp_Atmos_prev = nature["AtmosphericTemperature"][t-1] if t > 0 else nature["AtmosphericTemperature"][0]
    Temp_Oceans_prev = nature["LowerOceansTemperature"][t-1] if t > 0 else nature["LowerOceansTemperature"][0]

    # === Emissions industrielles (production-based, avec facteur mu_energy par secteur)
    industrial_emissions = 0.0
    epsilon_V = params["epsilon_V"]
    epsilon_B = params["epsilon_B"]
    emissions_share = params["emissions_share"]

    for firm in firms:
        # intensité énergétique de la firme (déjà sectorisée par prod_intensity_factor)
        energy_intensity = compute_firm_energy_intensity(firm, t, epsilon_V, epsilon_B)

        production = firm["Production"][t] if t < len(firm["Production"]) else 0.0
        energy_needed = energy_intensity * production

        green_cap = firm["GreenCapital"][t] if t < len(firm["GreenCapital"]) else 0
        brown_cap = firm["BrownCapital"][t] if t < len(firm["BrownCapital"]) else 0
        total_cap = green_cap + brown_cap
        green_ratio = green_cap / total_cap if total_cap > 0 else 0.0

        renewable_share = 1 / (1 + 1/(0.54 * green_ratio)) if green_ratio > 0 else 0.0
        renewable_energy = renewable_share * energy_needed
        non_renewable_energy = energy_needed - renewable_energy

        industrial_emissions += emissions_share * non_renewable_energy

    nature["IndustrialEmissions"][t] = float(industrial_emissions)

    # === Emissions usage des sols (inchangé)
    land_use_change = params.get("land_use_change", 0.01)
    if t == 0:
        nature["LandUseEmissions"][t] = 0.0
    else:
        nature["LandUseEmissions"][t] = float(nature["LandUseEmissions"][t-1] * (1 - land_use_change))

    # Total (production-based seulement)
    nature["TotalEmissions"][t] = float(nature["IndustrialEmissions"][t] + nature["LandUseEmissions"][t])

    # === CO2 box model (inchangé)
    phi_11 = params["phi_11"]; phi_21 = params["phi_21"]; phi_12 = params["phi_12"]
    phi_22 = params["phi_22"]; phi_32 = params["phi_32"]; phi_23 = params["phi_23"]; phi_33 = params["phi_33"]

    nature["AtmosphericCO2Concentration"][t] = float(nature["TotalEmissions"][t] + phi_11 * CO2_prev + phi_21 * Biosphere_CO2_prev)
    nature["BiosphereCO2Concentration"][t]   = float(phi_12 * CO2_prev + phi_22 * Biosphere_CO2_prev + phi_32 * LowerOceans_CO2_prev)
    nature["LowerOceansCO2Concentration"][t] = float(phi_23 * Biosphere_CO2_prev + phi_33 * LowerOceans_CO2_prev)

    # === Forçage radiatif (inchangé)
    RF_2CO2 = params["RF_2CO2"]
    CO2_preindustrial = params["AtmosphericPreIndustrialCO2Concentration"]
    ratio = nature["AtmosphericCO2Concentration"][t] / CO2_preindustrial if CO2_preindustrial > 0 else 1.0
    radiative_forcing = math.log(RF_2CO2 * ratio) / math.log(2) if ratio > 0 else 0.0
    nature["RadiativeForcing"] = radiative_forcing

    # === Températures (inchangé)
    CS = params["CS"]; t_1 = params["t_1"]; t_2 = params["t_2"]; RF = radiative_forcing
    part = RF - (RF_2CO2 / CS)
    temp_inter = part * Temp_Atmos_prev - t_2 * (Temp_Atmos_prev - Temp_Oceans_prev)
    nature["AtmosphericTemperature"][t] = t_1 * temp_inter

    t_3 = params["t_3"]
    nature["LowerOceansTemperature"][t] = Temp_Oceans_prev + t_3 * (Temp_Atmos_prev - Temp_Oceans_prev)

    # === Dommages (inchangé)
    eta_1 = params["eta_1"]; eta_2 = params["eta_2"]; eta_3 = params["eta_3"]
    T_at = nature["AtmosphericTemperature"][t]
    damages = eta_1 * T_at + eta_2 * (T_at ** 2) + eta_3 * (T_at ** 6.754)
    nature["DamagesFunction"][t] = 1 - 1 / (1 + damages)

def ksh_get(param_name, sid, ksh_sector_params):
    d = ksh_sector_params.get(param_name, {})
    return d.get(sid, list(d.values())[0] if d else 1.0)

# Déprécie K_shared et ajoute les mises en service avec les lags sectoriels
def roll_shared_capital(t, K_shared, I_pub_shared_by_sector, ksh_sector_params):
    if t == 0:
        return
    for s in range(1, 7):
        delta = ksh_get("delta", s, ksh_sector_params)
        lag = ksh_get("build_lag", s, ksh_sector_params)
        capex_per_unit = ksh_get("capex_per_unit", s, ksh_sector_params)
        prevK = K_shared[s][t-1]
        addK = 0.0
        if t - lag >= 0:
            addK = I_pub_shared_by_sector[s][t - lag] / max(1e-9, capex_per_unit)
        K_shared[s][t] = max(0.0, (1.0 - delta) * prevK + addK)

# Capacité publique dispo (en unités de service) et valeur au prix administré
def shared_capacity_value_at_t(sid, t, K_shared, ksh_sector_params, admin_factor,
                               price_ag_list, price_ener_list, price_hous_list,
                               price_trans_list, price_ind_list, price_tech_list):
    def sector_price_at_t(sid, t):
        maps = {1:price_ag_list,2:price_ener_list,3:price_hous_list,4:price_trans_list,5:price_ind_list,6:price_tech_list}
        L = maps.get(sid, None)
        if not L: return 1.0
        return L[t] if t < len(L) else L[-1]
    phi = ksh_get("phi_service", sid, ksh_sector_params)
    Kt = K_shared[sid][t]
    units = phi * Kt
    p_priv = sector_price_at_t(sid, t)
    p_admin = admin_factor * p_priv
    value_admin = units * p_admin
    return units, value_admin, p_admin

def seed_initial_shared_capital_from_private0(firm_list, K_shared, seed_ratio_by_sector, scenario_name, PostGrowthActive):
    # Active uniquement pour post_growth (et si le scénario est bien activé dès t=0)
    if scenario_name != "post_growth" or not PostGrowthActive or PostGrowthActive[0] != 1:
        return
    # Agrège K privé (Green+Brown) à t=0 par secteur
    sector_private_capital0 = {s: 0.0 for s in range(1,7)}
    for f in firm_list:
        sid = f.get("IdSector", 0)
        if sid not in sector_private_capital0: 
            continue
        g0 = f.get("GreenCapital",[0.0])[0] if f.get("GreenCapital") else 0.0
        b0 = f.get("BrownCapital",[0.0])[0] if f.get("BrownCapital") else 0.0
        sector_private_capital0[sid] += max(0.0, g0 + b0)

    # K_shared initial : ratio * K_privé(0) (cohérent avec ton paysage sectoriel)
    for s in range(1,7):
        K0 = seed_ratio_by_sector.get(s, 0.0) * sector_private_capital0[s]
        # sécurités
        if not np.isfinite(K0): K0 = 0.0
        K_shared[s][0] = float(max(0.0, K0))

# === [NOUVEAU] Part socialisée “par scénario” et “par secteur” ===
def social_share_for_sector(t, sector_id, scenario_name, PostGrowthActive, social):
    if scenario_name != "post_growth" or len(PostGrowthActive) <= t or PostGrowthActive[t] != 1:
        return 0.0
    target = social["target_share"].get(sector_id, 0.0)
    # rampe linéaire vers la cible
    share = target * min(1.0, max(0.0, t / max(1, social["ramp_periods"])))
    return max(0.0, min(1.0, share))

def apply_socialization_with_capacity(
    t,
    scenario_name,
    sector_revenue_market,
    firm_list,
    PostGrowthActive,
    social,
    pg_cons,
    K_shared,
    ksh_sector_params,
    I_pub_shared_by_sector,
    caproom_left,
    ksh_budget,
):
    """
    Socialisation avec support matériel :

    - la part socialisée visée est donnée par social_share_for_sector(...)
    - la part effectivement socialisée est bornée par la capacité publique K_shared,
      transformée en services via shared_capacity_value_at_t(...)
    - les recettes liées à la partie socialisée sont ventilées vers les firmes
      via allocate_shared_revenue_by_sector.

    Retourne :
      - sector_after_social : dict {NomSecteur -> dépense ménages (marché + ticket modérateur)}
      - G_socialized_t      : dépense publique de subvention (part non payée par les ménages)
    """

    # Si on n'est pas en post_growth (ou pas activé à t), aucune socialisation
    if scenario_name != "post_growth" or PostGrowthActive is None \
       or t >= len(PostGrowthActive) or PostGrowthActive[t] != 1:
        return dict(sector_revenue_market), 0.0

    sector_after_social = {}
    G_socialized_t = 0.0

    pg = pg_cons or {}
    sigma_target = pg.get("sigma_target", {}) or {}
    alpha_admin = pg.get("alpha_admin", {}) or {}
    copay_dict = pg.get("copay", {}) or {}

    # Mapping id -> nom de secteur (doit matcher compute_sector_revenue_base)
    name_by_id = {
        1: "Agriculture",
        2: "Energy",
        3: "Housing",
        4: "Transport",
        5: "Industry",
        6: "Technology",
    }

    hysteresis = float(social.get("hysteresis", 0.10) if social else 0.10)

    for sid, sname in name_by_id.items():
        base_val = float(sector_revenue_market.get(sname, 0.0))

        # Secteur absent ou sans demande -> on recopie
        if base_val <= 0.0:
            sector_after_social[sname] = base_val
            continue

        # Secteur non ciblé dans pg_cons -> pas de socialisation
        if float(sigma_target.get(sid, 0.0)) <= 0.0:
            sector_after_social[sname] = base_val
            continue

        # Part socialisée visée (rampe temporelle)
        share_target = social_share_for_sector(
            t, sid, scenario_name, PostGrowthActive, social
        )
        if share_target <= 0.0:
            sector_after_social[sname] = base_val
            continue

        # Paramètres : prix administré et ticket modérateur
        admin_mult = float(alpha_admin.get(sid, social.get("admin_price_factor", 0.60)))
        copay = float(copay_dict.get(sid, 0.0))
        admin_mult = max(0.0, admin_mult)
        copay = min(1.0, max(0.0, copay))

        # Capacité publique au t (en valeur au prix administré)
        _units_cap, cap_val_admin, p_admin = shared_capacity_value_at_t(
            sid, t, K_shared, ksh_sector_params,
            admin_mult,
            None, None, None, None, None, None  # prix sectoriels détaillés : placeholder
        )

        # Demande socialisée visée, convertie au prix admin
        desired_val_admin = base_val * share_target * admin_mult

        # Partie effectivement socialisée = bornée par la capacité publique
        used_val_admin = min(desired_val_admin, cap_val_admin)

        # Part effectivement socialisée (≈ en volume)
        if base_val > 1e-12 and admin_mult > 0:
            realized_share = used_val_admin / (admin_mult * base_val)
        else:
            realized_share = 0.0
        realized_share = max(0.0, min(share_target, realized_share))

        # Dépense des ménages :
        # - reste de marché au prix privé
        # - ticket modérateur sur la partie socialisée (au prix admin)
        private_val = base_val * (1.0 - realized_share)
        copay_val = copay * used_val_admin
        sector_after_social[sname] = private_val + copay_val

        # Dépense publique = subvention = (1 - copay) * valeur socialisée au prix admin
        subsidy = (1.0 - copay) * used_val_admin
        G_socialized_t += subsidy

        # Allocation de la recette socialisée (subvention + copay) vers les firmes du secteur
        sector_firms = [f for f in firm_list if f.get("IdSector") == sid]
        _allocated = allocate_shared_revenue_by_sector(
            t, sector_firms, used_val_admin, hysteresis
        )

        # Pour l'instant, on laisse I_pub_shared_by_sector[sid][t] = 0.0.
        # La montée en capacité publique par CAPEX sera modélisée plus tard,
        # en cohérence SFC. K_shared vient uniquement du semis de t=0.

    # S'il reste des secteurs dans sector_revenue_market qui ne sont pas dans name_by_id
    for sname, val in sector_revenue_market.items():
        if sname not in sector_after_social:
            sector_after_social[sname] = float(val)

    return sector_after_social, G_socialized_t

# === [NOUVEAU] Score de priorité pour contrats "shared" (simple, robuste) ===
def firm_shared_score(f, t):
    # 1) verdissement (↑ mieux)
    g = f.get("GreenCapital", [])
    b = f.get("BrownCapital", [])
    gv = g[t] if len(g) > t else (g[-1] if g else 0.0)
    bv = b[t] if len(b) > t else (b[-1] if b else 0.0)
    green_ratio = (gv / (gv + bv)) if (gv + bv) > 0 else 0.0

    # 2) levier (↓ mieux)
    lev_list = f.get("LeverageFirm", [])
    lev = lev_list[t] if len(lev_list) > t else (lev_list[-1] if lev_list else 1.0)
    lev_term = 1.0 / (1.0 + max(0.0, lev))

    # 3) fiabilité (variance de production passée, ↓ mieux)
    prod = f.get("Production", [])
    if len(prod) >= 4:
        window = prod[max(0, t-3):t+1] if t > 0 else prod[:1]
        var = float(np.var(window)) if window else 0.0
    else:
        var = 0.0
    rel_term = 1.0 / (1.0 + var)

    # pondération simple
    return 0.5 * green_ratio + 0.3 * lev_term + 0.2 * rel_term

# === [NOUVEAU] Allocation de la part "shared" par secteur ===
def allocate_shared_revenue_by_sector(t, sector_firms, shared_value, hysteresis):
    """
    Répartit le montant shared_value entre les firmes d'un secteur selon le score.
    Met à jour f["AlphaShared"][t] et f["SharedRevenue"][t].
    Renvoie le total effectivement alloué (peut être < shared_value si manque de capacité privée).
    """
    if shared_value <= 0 or not sector_firms:
        for f in sector_firms:
            # assure la longueur
            while len(f["AlphaShared"]) <= t: f["AlphaShared"].append(0.0)
            while len(f["SharedRevenue"]) <= t: f["SharedRevenue"].append(0.0)
        return 0.0

    # Scores
    scored = []
    for f in sector_firms:
        s = firm_shared_score(f, t)
        scored.append((s, f))
    scored.sort(reverse=True, key=lambda x: x[0])

    # Allocation proportionnelle aux scores (normalisée)
    total_score = sum(max(0.0, s) for s, _ in scored)
    if total_score <= 0:
        # split égal si tous les scores sont nuls
        per = shared_value / len(scored)
        allocated = 0.0
        for _, f in scored:
            while len(f["AlphaShared"]) <= t: f["AlphaShared"].append(0.0)
            while len(f["SharedRevenue"]) <= t: f["SharedRevenue"].append(0.0)
            f["SharedRevenue"][t] = per
            # AlphaShared rétro-inférée comme part des ventes (indicative)
            f["AlphaShared"][t] = min(1.0, max(0.0, f["AlphaShared"][t-1] if t>0 and len(f["AlphaShared"])>t-1 else 0.0))
            allocated += per
        return allocated

    allocated = 0.0
    for s, f in scored:
        share = max(0.0, s) / total_score
        chunk = share * shared_value
        while len(f["AlphaShared"]) <= t: f["AlphaShared"].append(0.0)
        while len(f["SharedRevenue"]) <= t: f["SharedRevenue"].append(0.0)

        # Hystérésis sur AlphaShared (si on l'utilise plus tard comme part de prod)
        prev = f["AlphaShared"][t-1] if t>0 and len(f["AlphaShared"])>t-1 else 0.0
        # ici on borne juste la variation, valeur indicative, la vraie saturation vient par la demande
        f["AlphaShared"][t] = max(0.0, min(1.0, prev + np.sign(chunk - prev) * hysteresis))
        f["SharedRevenue"][t] = chunk
        allocated += chunk

    return allocated


def plot_atmospheric_temperature(results_df):
    """
    Trace l'évolution de la température atmosphérique
    pour chaque scénario de politique.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=results_df,
        x="Period",
        y="AtmosphericTemperature",
        hue="ScenarioName",
        estimator="mean",
        ci="sd"
    )
    plt.title("Évolution de la température atmosphérique selon les scénarios")
    plt.xlabel("Période")
    plt.ylabel("Température atmosphérique (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def gini_coefficient(x):
    import numpy as np
    sorted_x = np.sort(np.array(x))
    n = len(x)
    cumx = np.cumsum(sorted_x)
    sum_x = cumx[-1]
    if sum_x == 0:
        return 0
    gini = (n + 1 - 2 * np.sum(cumx) / sum_x) / n
    return gini



def initialize_firm_products(firms, p_green=0.1):
    """
    Initialise les produits des firmes avec hétérogénéité.
    p_green = part des firmes qui démarrent déjà avec Portfolio=2 (techno verte).
    """
    for f in firms:
        # Produit 1 (brun)
        prod1 = {
            "IdProduct": 1,
            "IsAdopted": [1],
            "X": [np.random.uniform(0.2, 0.4)],     # qualité
            "Eff": [np.random.uniform(0.1, 0.3)],   # efficacité
            "Tox": [np.random.uniform(0.5, 0.9)],   # toxicité
            "Bio": [np.random.uniform(0.5, 0.9)],   # empreinte biologique
            "Price": [np.random.uniform(0.8, 1.2)], # prix
            "Sales_Prod": [0]
        }

        # Produit 2 (vert)
        prod2 = {
            "IdProduct": 2,
            "IsAdopted": [0],  # par défaut non adopté
            "X": [np.random.uniform(0.3, 0.6)],
            "Eff": [np.random.uniform(0.2, 0.5)],
            "Tox": [np.random.uniform(0.1, 0.4)],
            "Bio": [np.random.uniform(0.1, 0.4)],
            "Price": [np.random.uniform(0.9, 1.3)],
            "Sales_Prod": [0]
        }

        # Par défaut Portfolio = 0 → seulement produit 1
        f["Products"] = [prod1, prod2]
        f["Portfolio"] = [0]

        # Introduire une fraction de pionniers
        if np.random.rand() < p_green:
            f["Portfolio"] = [2]               # uniquement produit 2
            f["Products"][1]["IsAdopted"] = [1]
            f["Products"][0]["IsAdopted"] = [0]

        # Initialiser les variables agrégées de la firme
        for var in ["X", "Eff", "Tox", "Bio", "Price"]:
            f[var] = []



def initialize_households_and_sales_from_struct(households_struct, firms, params, num_sectors=6,
                                               rural_prefs=None, urban_prefs=None):
    """
    Initialise les ménages à partir de households_struct,
    ajoute les variables nécessaires (ProdUsed, ReservationPrice, etc.)
    et attribue des ventes initiales aux firmes.
    """
    household_list = []

    # Chaque ménage a une "phase" d'achat par secteur (5,6), pour désynchroniser (éviter les à-coups)
    for h in household_list:
        h.setdefault("PurchasePhase", {})
        for sid in (5, 6):
            n = purchase_freq.get(sid, 1)
            h["PurchasePhase"][sid] = random.randint(0, max(0, n-1))

    for h in households_struct:
        income_init = random.uniform(0.2, 0.45)
        savings_init = random.uniform(0.2, 0.45)

        # Préférences selon territoire
        prefs = rural_prefs if h["IdTerritory"] == 0 else urban_prefs
        prefscons = np.random.rand(4)
        prefscons /= prefscons.sum()
        if h["IdTerritory"] == 0:     # 0 = rural
            ptc_di = 0.713            # 71.3 %
        else:                         # 1 = urbain
            ptc_di = 0.717            # 71.7 %
        hh = {
            # === Infos de households_struct conservées ===
            "Status": h["Status"],
            "SkillStatus": h["SkillStatus"],
            "IdTerritory": h["IdTerritory"],
            "IdEmployer": h["IdEmployer"],
            "AgPref": h["AgPref"],
            "EnerPref": h["EnerPref"],
            "HousPref": h["HousPref"],
            "TransPref": h["TransPref"],
            "IndPref": h["IndPref"],
            "TCPref": h["TCPref"],

            # === Variables économiques ===
            "FirmID": [-1],
            "DisposableIncome": [income_init],
            "Income": [income_init],
            "FinancialIncome": [0],
            "Red": [0.05],
            "TaxesInd": [0],
            "HouseholdDebtService": [],
            "LoanRateCons": [0.03],
            "HouseholdNewDebt": [],
            "HouseholdTotalDebt": [0.01],
            "BaseConsumption": [],
            "Consumption": [],
            "Savings": [savings_init],
            "DepositInterestRate": [0.01],
            "TaxRate": [0.2],
            "PropensityToConsumeDI": [ptc_di],
            "PropensityToConsumeSavings": [0.01],
            "NeedsIndex": [100.0],
            "GrNeedsIndex": [],
            "VoteDecision": [random.randint(0, 1)],
            "DistanceToJob": random.uniform(3, 7) if h["IdTerritory"] == 0 else random.uniform(0.5, 3),
            "a": prefscons[0], "b": prefscons[1], "c": prefscons[2], "d": prefscons[3],
            "e": np.random.uniform(0.5, 1.5),

            # === Variables pour l’innovation ===
            "Reservation_Price": np.random.uniform(100, 1000),
            "Reservation_X": np.random.uniform(0, 100),
            "RebuyProb": [0.8],
            "Regulation_Threat_Customer": [0],
            "Product_Type": [1],
            "ProdUsed": {sector_id: [1] for sector_id in range(1, num_sectors + 1)} , # tous sur produit 1 au départ
            "SupplierBySector": {sector_id: [-1] for sector_id in range(1, num_sectors + 1)}

        }

        # juste avant household_list.append(hh)
        hh["PurchasePhase"] = {}
        for sid in (5, 6):
            n = purchase_freq.get(sid, 1)
            hh["PurchasePhase"][sid] = random.randint(0, max(0, n-1))

        household_list.append(hh)

    """
    # === AVANT la boucle des périodes ===
    CarbonTaxActive, TransitionActive, PostGrowthActive = [], [], []

    def update_policy_activation(t, household_list, policy_list, scenario_triggered):
        if t == 0:
            policy_list.append(0)
        elif scenario_triggered and t in [5, 10, 15, 20, num_periods - 1]:
            # On vote : si tous ont un vote à t, on tranche, sinon on prolonge
            if all(len(h.get("VoteDecision", [])) > t for h in household_list):
                nb_pro = sum(1 for h in household_list if h["VoteDecision"][t] == 1)
                policy_list.append(1 if nb_pro > num_households // 2 else policy_list[t-1])
            else:
                policy_list.append(policy_list[t - 1])
        else:
            policy_list.append(policy_list[t - 1])

    # Construire les drapeaux pour tout l’horizon, en fonction du scénario choisi
    for tt in range(num_periods):
        update_policy_activation(tt, household_list, CarbonTaxActive,  scenario_name == "carbon_tax_only")
        update_policy_activation(tt, household_list, TransitionActive, scenario_name == "transition_mix")
        update_policy_activation(tt, household_list, PostGrowthActive, scenario_name == "post_growth")
    """


    # === Mise à jour des firmes : ventes initiales ===
    for f in firms:
        f["Sales"] = [0]
        f["Sales_Prod"] = {1: [0], 2: [0]}

    # === Attribution des ventes initiales (cohérente produit/fournisseur) ===
    for hh in household_list:
        for sector_id in range(1, num_sectors + 1):
            chosen_prod = hh["ProdUsed"][sector_id][0]  # 1 ou 2
            candidates = [f for f in firms if f.get("IdSector") == sector_id]
            if not candidates:
                continue
            chosen_firm = random.choice(candidates)

            # NEW: on mémorise le fournisseur
            hh["SupplierBySector"][sector_id][0] = chosen_firm["FirmID"]

            # ventes : firm + produit
            chosen_firm["Sales"][-1] += 1
            chosen_firm["Sales_Prod"].setdefault(chosen_prod, [0])
            chosen_firm["Sales_Prod"][chosen_prod][-1] += 1

    return household_list, firms

   
def safe_get_last(lst, t):
    """Retourne lst[t] si dispo, sinon la dernière valeur existante, ou 0 si vide."""
    if len(lst) == 0:
        return 0
    return lst[t] if t < len(lst) else lst[-1]

def update_firm_characteristics_from_products(firms, t):
    """
    Met à jour les caractéristiques agrégées (X, Eff, Tox, Bio, Price)
    de chaque firme en fonction de ses produits actifs.
    - Portfolio = 0 → produit 1 seul
    - Portfolio = 2 → produit 2 seul
    - Portfolio = 1 → moyenne pondérée par ventes des deux produits
    """

    for f in firms:
        portfolio = f.get("Portfolio", [0])[-1]
        # Initialiser les champs si nécessaire
        for var in ["X", "Eff", "Tox", "Bio", "Price"]:
            if var not in f:
                f[var] = []

        # Cas 1 : produit 1 seul
        if portfolio == 0:
            prod = f["Products"][0]
            f["X"].append(safe_get_last(prod["X"], t))
            f["Eff"].append(safe_get_last(prod["Eff"], t))
            f["Tox"].append(safe_get_last(prod["Tox"], t))
            f["Bio"].append(safe_get_last(prod["Bio"], t))
            f["Price"].append(safe_get_last(prod["Price"], t))

        # Cas 2 : produit 2 seul
        elif portfolio == 2:
            prod = f["Products"][1]
            f["X"].append(safe_get_last(prod["X"], t))
            f["Eff"].append(safe_get_last(prod["Eff"], t))
            f["Tox"].append(safe_get_last(prod["Tox"], t))
            f["Bio"].append(safe_get_last(prod["Bio"], t))
            f["Price"].append(safe_get_last(prod["Price"], t))

        # Cas 3 : les deux produits
        elif portfolio == 1:
            prod1, prod2 = f["Products"][0], f["Products"][1]

            sales1 = safe_get_last(prod1.get("Sales_Prod", []), t)
            sales2 = safe_get_last(prod2.get("Sales_Prod", []), t)
            total_sales = sales1 + sales2

            if total_sales > 0:
                w1, w2 = sales1 / total_sales, sales2 / total_sales
            else:
                w1 = w2 = 0.5  # pondération égale si aucune vente

            f["X"].append(w1 * safe_get_last(prod1["X"], t) + w2 * safe_get_last(prod2["X"], t))
            f["Eff"].append(w1 * safe_get_last(prod1["Eff"], t) + w2 * safe_get_last(prod2["Eff"], t))
            f["Tox"].append(w1 * safe_get_last(prod1["Tox"], t) + w2 * safe_get_last(prod2["Tox"], t))
            f["Bio"].append(w1 * safe_get_last(prod1["Bio"], t) + w2 * safe_get_last(prod2["Bio"], t))
            f["Price"].append(w1 * safe_get_last(prod1["Price"], t) + w2 * safe_get_last(prod2["Price"], t))

        # Cas par défaut
        else:
            f["X"].append(0)
            f["Eff"].append(0)
            f["Tox"].append(0)
            f["Bio"].append(0)
            f["Price"].append(1.0)

def is_buyer_this_period(h, t, sid, purchase_freq):
    n = purchase_freq.get(sid, 1)
    if n <= 1:
        return True
    phase = h.get("PurchasePhase", {}).get(sid, 0)
    return (t % n) == phase

def buyers_share_at_t(households, t, sid, purchase_freq):
    elig = sum(1 for h in households if is_buyer_this_period(h, t, sid, purchase_freq))
    tot  = max(1, len(households))
    return elig / tot

def used_share_for_sector(t, sid, used_params):
    target = used_params["target_used_share"].get(sid, 0.0)
    ramp   = max(1, used_params.get("ramp_periods", 8))
    phase  = min(1.0, max(0.0, t / ramp))
    return target * phase


def update_firm_prices_from_sector_lists(
    firms, t,
    price_ag_list, price_ener_list,
    price_hous_list, price_trans_list,
    price_ind_list, price_tech_list
):
    """
    Met à jour les prix des firmes à partir des listes de prix sectoriels.
    Associe automatiquement le bon prix en fonction de IdSector.
    """

    for f in firms:
        sector_id = f["IdSector"]

        if sector_id == 1:   # Agriculture
            price = price_ag_list[t]
        elif sector_id == 2: # Énergie
            price = price_ener_list[t]
        elif sector_id == 3: # Housing
            price = price_hous_list[t]
        elif sector_id == 4: # Transport
            price = price_trans_list[t]
        elif sector_id == 5: # Industrie
            price = price_ind_list[t]
        elif sector_id == 6: # Technologie & Communication
            price = price_tech_list[t]
        else:
            price = 1.0  # valeur par défaut si erreur

        # Ajout du champ Price si inexistant
        if "Price" not in f:
            f["Price"] = []

        # Ajout du prix pour la période t
        f["Price"].append(price)

# === UTILITAIRES ===

def draw_with_prob(weights):
    """Tirage proportionnel aux poids (équivalent RNDDRAW LSD)."""
    weights = np.array(weights, dtype=float)
    if weights.sum() <= 0:
        return None
    probs = weights / weights.sum()
    return np.random.choice(len(weights), p=probs)


# === PROCESSUS REBUY ===

def try_rebuy(hh, prev_firm_id, sector_id, firms, t, params):
    """
    Retourne True si le ménage garde son fournisseur précédent (prev_firm_id),
    en vérifiant que la firme existe, est vivante et dans le bon secteur.
    """
    if t == 0 or prev_firm_id in (None, -1):
        return False

    firm = next((f for f in firms
                 if f.get("FirmID") == prev_firm_id
                 and f.get("IdSector") == sector_id
                 and (f.get("Dead", [0])[-1] if f.get("Dead") else 0) == 0), None)
    if firm is None:
        return False

    # Préférences du ménage (inchangées)
    a, b, c, d = hh["a"], hh["b"], hh["c"], hh["d"]
    total_pref = a + b + c + d
    pa, pb, pc, pd = a / total_pref, b / total_pref, c / total_pref, d / total_pref

    r = np.random.rand()
    # seuils “mémoire” par ménage (grâce aux champs *_prev)
    if r <= pa:
        return firm["X"][-1] >= hh.get("Max_X_prev", 0) * np.random.uniform(0.8, 1.2)
    elif r <= pa + pb:
        return firm["Price"][-1] <= hh.get("Min_Price_prev", 0) * np.random.uniform(0.8, 1.2)
    elif r <= pa + pb + pc:
        return firm["Tox"][-1] <= hh.get("Min_Tox_prev", 0) * np.random.uniform(0.8, 1.2)
    else:
        return firm["Bio"][-1] <= hh.get("Min_Bio_prev", 0) * np.random.uniform(0.8, 1.2)


# === PROCESSUS PURCHASE ===

def safe_get(lst, t, default=0):
    """Retourne lst[t] si possible, sinon la dernière valeur ou un défaut si liste vide."""
    return lst[t] if t < len(lst) else (lst[-1] if lst else default)


def purchase_new(hh, sector_id, firms, t, params):
    """
    Sélectionne une nouvelle firme dans le secteur, en fonction des préférences du ménage.
    Retourne l'IdFirm choisi ou None si aucun.
    """

    # Préférences du ménage
    a, b, c, d, e = hh.get("a", 1), hh.get("b", 1), hh.get("c", 1), hh.get("d", 1), hh.get("e", 1)
    total_pref = a + b + c + d
    if total_pref == 0:
        total_pref = 1  # éviter une division par zéro
    pa, pb, pc, pd = a / total_pref, b / total_pref, c / total_pref, d / total_pref

    # Tirage sur la dimension de décision
    r = np.random.rand()

    # Variables de seuil du ménage
    reservation_price = hh.get("Reservation_Price", 100)
    reservation_x = hh.get("Reservation_X", 0.0)
    reg_threat = safe_get(hh.get("Regulation_Threat_Customer", [0]), t, 0)

    utilities = []
    candidate_firms = []

    for firm in firms:
        if firm.get("IdSector", -1) != sector_id or safe_get(firm.get("Dead", [0]), t, 0) == 1:
            continue

        X = safe_get(firm.get("X", [0]), t, 0)
        price = safe_get(firm.get("Price", [1.0]), t, 1.0)
        tox = safe_get(firm.get("Tox", [1.0]), t, 1.0)
        bio = safe_get(firm.get("Bio", [1.0]), t, 1.0)
        ms = safe_get(firm.get("MS", [1.0]), t, 1.0)
        adopted = safe_get(firm.get("IsAdopted", [1] * len(firm.get("X", []))), t, 1)

        if adopted == 0:
            continue

        error = np.random.uniform(0.8, 1.2)

        if price > reservation_price or X < reservation_x:
            util = 0
        else:
            if r <= pa:
                util = (X * error - params["A"]) * (ms + np.random.uniform(0, 0.1)) ** e
            elif r <= pa + pb:
                util = (params["B"] - price * error) * (ms + np.random.uniform(0, 0.1)) ** e
            elif r <= pa + pb + pc:
                util = (params["C"] - tox * error) * (ms + np.random.uniform(0, 0.1)) ** e
            else:
                util = (params["D"] - bio * error) * (ms + np.random.uniform(0, 0.1)) ** e

            portfolio = firm.get("Portfolio", [0])
            if safe_get(portfolio, t, 0) == 0:
                util *= (1 - reg_threat)


        utilities.append(max(util, 0))
        candidate_firms.append(firm)

    if not utilities or sum(utilities) == 0:
        return None

    idx = draw_with_prob(utilities)
    return candidate_firms[idx].get("IdFirm", None)


# === PROCESSUS GLOBAL PAR SECTEUR ===

def update_household_sector_choice(hh, sector_id, firms, t, params):
    """
    Met à jour le fournisseur (SupplierBySector) et compte les ventes.
    ProdUsed (id produit) reste un autre registre, non confondu avec le fournisseur.
    """
    prev_firm_id = hh["SupplierBySector"][sector_id][t-1] if t > 0 else -1

    # 1) tenter rebuy
    if try_rebuy(hh, prev_firm_id, sector_id, firms, t, params):
        chosen_firm_id = prev_firm_id
    else:
        # 2) nouveau choix (utilitaires inchangés)
        chosen_firm_id = purchase_new(hh, sector_id, firms, t, params)

    if chosen_firm_id is None:
        chosen_firm_id = -1

    # Historisation fournisseur
    if len(hh["SupplierBySector"][sector_id]) > t:
        hh["SupplierBySector"][sector_id][t] = chosen_firm_id
    else:
        hh["SupplierBySector"][sector_id].append(chosen_firm_id)

    # Compter la vente côté firme/produit
    if chosen_firm_id != -1:
        firm = next((f for f in firms if f.get("FirmID") == chosen_firm_id and f.get("IdSector") == sector_id), None)
        if firm:
            # init Sales/produits si besoin
            firm.setdefault("Sales", [])
            while len(firm["Sales"]) <= t: firm["Sales"].append(0)
            firm["Sales"][t] += 1

            # choisir le produit actif (portfolio) pour créditer Sales_Prod
            active_pid = 2 if (firm.get("Portfolio", [0])[-1] == 2) else 1
            if "Products" in firm:
                for p in firm["Products"]:
                    if p.get("IdProduct") == active_pid:
                        p.setdefault("Sales_Prod", [])
                        while len(p["Sales_Prod"]) <= t: p["Sales_Prod"].append(0)
                        p["Sales_Prod"][t] += 1
                        break

# === PROCESSUS GLOBAL TOUS SECTEURS ===

def update_household_consumption(households, firms, t, params, num_sectors=6):
    """
    Met à jour les choix de fournisseur par ménage/secteur, une seule fois par période et par ménage.
    Garantit que les listes ont bien un index t (pas de trous).
    """
    for hh in households:
        # idempotence par ménage
        if hh.get("_last_written_t", -1) == t:
            continue

        # assurer index t pour les registres clés (évite CTot=0 si une branche saute)
        for sid in range(1, num_sectors+1):
            hh["ProdUsed"].setdefault(sid, [])
            hh["SupplierBySector"].setdefault(sid, [])
            while len(hh["ProdUsed"][sid]) <= t:          hh["ProdUsed"][sid].append(hh["ProdUsed"][sid][-1] if hh["ProdUsed"][sid] else 1)
            while len(hh["SupplierBySector"][sid]) <= t:  hh["SupplierBySector"][sid].append(hh["SupplierBySector"][sid][-1] if hh["SupplierBySector"][sid] else -1)

        # mises à jour par secteur
        for sector_id in range(1, num_sectors+1):
            update_household_sector_choice(hh, sector_id, firms, t, params)

        hh["_last_written_t"] = t

if "antiobs" not in globals():
    antiobs = {
        "target_drop_new": {1: 0.6, 2: 0.5, 3: 0.4, 4: 0.6, 5: 0.5, 6: 0.7},
        "ramp_periods": 4,
        "repair_price_factor": 0.35,
        "repair_labour_intensity": 1.30,
        "gamma": 1.4,             # accélère en début de rampe
        "min_multiplier": 0.05     # on tolère 5% de "neuf" incompressible
    }

# --- [SUFFISANCE] Fin de l’obsolescence : multiplicateur de valeur "neuf" restante ---
def antiobs_multiplier_new_value(t, sid, scenario_name, PostGrowthActive, antiobs):
    """
    Part de valeur 'neuf/remplacement' qui subsiste à t (1.0 = pas d'effet).
    On rend l'effet plus fort via (i) une rampe basée sur les périodes *effectivement actives*,
    (ii) une courbe non-linéaire (gamma > 1 accélère la baisse),
    (iii) un plancher min_multiplier pour éviter de retomber à 0.
    """
    # gardes de sécurité
    if not antiobs or scenario_name != "post_growth":
        return 1.0
    if PostGrowthActive is None or t >= len(PostGrowthActive) or PostGrowthActive[t] != 1:
        return 1.0

    drop = float(antiobs.get("target_drop_new", {}).get(sid, 0.0))          # ex: 0.6 → -60% du "neuf"
    ramp = max(1, int(antiobs.get("ramp_periods", 6)))                      # plus court = plus violent
    gamma = float(antiobs.get("gamma", 1.4))                                 # >1 accélère la baisse
    min_mult = float(antiobs.get("min_multiplier", 0.05))                    # 5% de "neuf" au plancher

    # nombre de périodes consécutives *actives* jusqu'à t (et pas t absolu)
    k = 0
    i = t
    while i >= 0 and PostGrowthActive[i] == 1:
        k += 1
        i -= 1

    phase = min(1.0, k / ramp)               # 0 → 1 le long de la rampe
    phased = phase ** gamma                   # accélération non linéaire
    raw = 1.0 - drop * phased                 # 1 → 1 - drop
    return max(min_mult, min(1.0, raw))

def pg_sigma(t, sid, scenario_name, PostGrowthActive, pg_cons):
    if scenario_name!="post_growth" or not PostGrowthActive or len(PostGrowthActive)<=t or PostGrowthActive[t]!=1:
        return 0.0
    target = pg_cons["sigma_target"].get(sid, 0.0)
    ramp   = max(1, pg_cons.get("ramp_periods", 8))
    phase  = min(1.0, max(0.0, t / ramp))
    return target * phase

def get_price_sector(sid, t, price_ag_list, price_ener_list, price_hous_list, price_trans_list, price_ind_list, price_tech_list):
    L = {1:price_ag_list,2:price_ener_list,3:price_hous_list,4:price_trans_list,5:price_ind_list,6:price_tech_list}.get(sid, None)
    if not L: return 1.0
    return L[t] if t < len(L) else L[-1]


# =========================================================
# A) BASE : agrégation conso → répartition aux firmes → profits (SANS modules)
# =========================================================
def compute_sector_revenue_base(t, household_list):
    """
    Renvoie un dict {secteur: valeur} construit comme AVANT (base + suppléments ménage),
    sans suffisance ni socialisation.
    """
    sectors   = ["Agriculture", "Energy", "Housing", "Transport", "Industry", "Technology"]
    supp_keys = ["SuppConsAg", "SuppConsEner", "SuppConsHous", "SuppConsTrans", "SuppConsInd", "SuppConsTC"]
    pref_keys = ["AgPref", "EnerPref", "HousPref", "TransPref", "IndPref", "TCPref"]

    sector_revenue = {s: 0.0 for s in sectors}
    for h in household_list:
        base_c = h["BaseConsumption"][t] if len(h.get("BaseConsumption", [])) > t else 0.0
        for i, s in enumerate(sectors):
            base_share = max(0.0, base_c * float(h.get(pref_keys[i], 0.0)))
            supp_share = max(0.0, (h[supp_keys[i]][t] if supp_keys[i] in h and len(h[supp_keys[i]]) > t else 0.0))
            sector_revenue[s] += base_share + supp_share

    return sector_revenue

# =========================================================
# B) Répartition sector_revenue → firmes + profits + ventes
# =========================================================
def distribute_revenue_and_compute_profits(t, firm_list, household_list, sector_revenue, CarbonTaxActive):
    sectors = ["Agriculture", "Energy", "Housing", "Transport", "Industry", "Technology"]

    # 1) Répartition aux firmes (égalitaire intra-secteur)
    firms_by_sector = {s: [] for s in sectors}
    for f in firm_list:
        sid = int(f.get("IdSector", 0))
        if 1 <= sid <= 6:
            firms_by_sector[sectors[sid - 1]].append(f)

    for s in sectors:
        firms = firms_by_sector[s]
        total_rev = float(sector_revenue.get(s, 0.0))
        if not firms:
            continue
        rev_per_firm = total_rev / len(firms) if len(firms) > 0 else 0.0
        for f in firms:
            f.setdefault("Revenue", [])
            while len(f["Revenue"]) <= t:
                f["Revenue"].append(0.0)
            f["Revenue"][t] = rev_per_firm

    # 2) Profits + taxe carbone explicite
    for f in firm_list:
        revenue  = f["Revenue"][t] if "Revenue" in f and len(f["Revenue"]) > t else 0.0
        firm_id  = f.get("FirmID", -1)

        # Masse salariale payée par la firme à t
        wages_paid = 0.0
        for h in household_list:
            if h.get("IdEmployer", -1) != firm_id:
                continue
            status = h.get("Status", 0)
            if status in [1, 2, 4, 5]:  # emplois marchands (bruns/verts, peu/qualifiés)
                income_list = h.get("Income", [])
                if len(income_list) > t:
                    wages_paid += float(income_list[t])

        # Intérêts payés sur dettes
        interest_paid = 0.0
        for key, rate_key in [("BrownLoans", "LoanInterestRate"),
                              ("GreenLoans", "GreenLoanInterestRate")]:
            loans = f.get(key, [])
            rates = f.get(rate_key, [])
            if len(loans) > t and len(rates) > t:
                interest_paid += float(loans[t]) * float(rates[t])

        # --- TAXE CARBONE SUR LE CAPITAL BRUN ---
        carbon_tax = 0.0
        if CarbonTaxActive and len(CarbonTaxActive) > t and CarbonTaxActive[t] == 1:
            brown_list = f.get("BrownCapital", [])
            if brown_list:
                brown_cap_t = brown_list[t] if t < len(brown_list) else brown_list[-1]
                carbon_tax_rate = 0.05  # même taux que dans le bloc d'investissement
                carbon_tax = carbon_tax_rate * max(0.0, float(brown_cap_t))

        # Profit net après taxe carbone
        profit = revenue - wages_paid - interest_paid - carbon_tax

        f.setdefault("Profits", [])
        while len(f["Profits"]) <= t:
            f["Profits"].append(0.0)
        f["Profits"][t] = profit

        # --- NOUVEAU : on stocke explicitement la taxe carbone payée ---
        f.setdefault("CarbonTaxPaid", [])
        while len(f["CarbonTaxPaid"]) <= t:
            f["CarbonTaxPaid"].append(0.0)
        f["CarbonTaxPaid"][t] = carbon_tax

        # Part de la taxe dans le chiffre d'affaires (utile pour diagnostics / comportements)
        f.setdefault("CarbonTaxShare", [])
        while len(f["CarbonTaxShare"]) <= t:
            f["CarbonTaxShare"].append(0.0)
        if revenue > 0.0:
            f["CarbonTaxShare"][t] = carbon_tax / revenue
        else:
            f["CarbonTaxShare"][t] = 0.0

# =========================================================
# Module SUFFISANCE : ajuste la demande des secteurs 5-6 et renvoie sector_revenue_market
# =========================================================
def apply_sufficiency_module(t, sector_revenue, household_list,
                             purchase_freq=None, used_params=None,
                             buyers_share_at_t=None,
                             used_share_for_sector=None,
                             antiobs_multiplier_new_value=None,
                             New_value_by_sector=None, Repair_value_by_sector=None):
    """
    Prend sector_revenue (dict) et retourne sector_revenue_market (dict) après :
    - achat épisodique (bshare)
    - anti-obsolescence (m_new)
    - marché de l'occasion (u_share, used_pf)
    Si helpers sont None, on applique des valeurs par défaut stables.
    Clamp systématique pour éviter les négatifs.
    """
    sectors = ["Agriculture", "Energy", "Housing", "Transport", "Industry", "Technology"]
    sector_revenue_market = dict(sector_revenue)  # copie
    purchase_freq = purchase_freq or {5: 8, 6: 4}
    used_params   = used_params   or {"used_price_factor": 0.5, "divert_to_repair_when_suppressed": 0.5}

    def clamp01(x):
        try:
            x = float(x)
        except Exception:
            x = 0.0
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    for sid, sname in enumerate(sectors, start=1):
        gross = float(sector_revenue.get(sname, 0.0))
        if gross <= 0.0 or sid not in (5, 6):
            sector_revenue_market[sname] = max(0.0, sector_revenue_market.get(sname, 0.0))
            continue

        # bshare
        if buyers_share_at_t is None:
            # défaut : fraction constante égale à 1/freq
            freq = max(1, int(purchase_freq.get(sid, 1)))
            bshare = 1.0 / float(freq)
        else:
            bshare = clamp01(buyers_share_at_t(household_list, t, sid, purchase_freq))

        allowed   = max(0.0, gross * bshare)
        suppressed = max(0.0, gross - allowed)

        # détour vers réparation
        divert_r = clamp01(used_params.get("divert_to_repair_when_suppressed", 0.5))
        add_repair_suppressed = suppressed * divert_r

        if Repair_value_by_sector is not None:
            while len(Repair_value_by_sector[sid]) <= t:
                Repair_value_by_sector[sid].append(0.0)
            Repair_value_by_sector[sid][t] += add_repair_suppressed

        # anti-obsolescence
        if antiobs_multiplier_new_value is None:
            m_new = 1.0  # neutre
        else:
            m_new = max(0.0, float(antiobs_multiplier_new_value(t, sid, globals().get("scenario_name", ""), globals().get("PostGrowthActive", []), globals().get("antiobs", {}))))
        new_after_antiobs = max(0.0, allowed * m_new)
        repair_from_antiobs = max(0.0, allowed - new_after_antiobs)

        # occasion
        if used_share_for_sector is None:
            u_share = 0.0  # pas d'occasion par défaut
        else:
            u_share = clamp01(used_share_for_sector(t, sid, used_params))

        used_pf = used_params.get("used_price_factor", 0.5)
        used_pf = clamp01(used_pf)

        used_paid     = max(0.0, new_after_antiobs * u_share * used_pf)
        used_discount = max(0.0, new_after_antiobs * u_share * (1.0 - used_pf))
        final_new_val = max(0.0, new_after_antiobs * (1.0 - u_share))

        # maj traces et revenus
        if Repair_value_by_sector is not None:
            Repair_value_by_sector[sid][t] += repair_from_antiobs + used_discount
        if New_value_by_sector is not None:
            while len(New_value_by_sector[sid]) <= t:
                New_value_by_sector[sid].append(0.0)
            New_value_by_sector[sid][t] = final_new_val

        sector_revenue_market[sname] = max(0.0, final_new_val + used_paid)

    return sector_revenue_market

# =========================================================
# Module SOCIALISATION : split C marché ↔ G socialisé (scénarios)
# =========================================================
def apply_socialization_module(t, sector_revenue_market,
                               PostGrowthActive=None, TransitionActive=None,
                               pg_cons=None, soft_factor_tm=0):
    """
    Prend sector_revenue_market et renvoie :
        sector_revenue_after_social (dict), G_socialized_t (float)
    Si aucun scénario actif, renvoie l'entrée telle quelle et G=0.
    """
    sectors = ["Agriculture", "Energy", "Housing", "Transport", "Industry", "Technology"]
    PostGrowthActive = PostGrowthActive or []
    TransitionActive = TransitionActive or []
    pg_cons = pg_cons or {}
    admin_price_default = 0.5

    is_pg = bool(PostGrowthActive[t]) if len(PostGrowthActive) > t else False
    is_tm = bool(TransitionActive[t]) if len(TransitionActive) > t else False
    if not is_pg and not is_tm:
        # rien à faire
        return dict(sector_revenue_market), 0.0

    social_share_pg = {
        "Agriculture": 0.30, "Energy": 0.60, "Housing": 0.60,
        "Transport": 0.60, "Industry": 0.00, "Technology": 0.00
    }
    sector_id_by_name = {
        "Agriculture": 1, "Energy": 2, "Housing": 3,
        "Transport": 4, "Industry": 5, "Technology": 6
    }

    out = dict(sector_revenue_market)
    G_socialized_t = 0.0

    def get_admin(sid):
        if "alpha_admin" in pg_cons:
            return float(pg_cons["alpha_admin"].get(sid, admin_price_default))
        return admin_price_default

    def get_copay(sid):
        if "copay" in pg_cons:
            return float(pg_cons["copay"].get(sid, 0.0))
        return 0.0

    for sname in sectors:
        base_val = float(sector_revenue_market.get(sname, 0.0))
        if base_val <= 0.0:
            out[sname] = 0.0
            continue

        sid = sector_id_by_name[sname]
        share_pg = social_share_pg.get(sname, 0.0)
        share = share_pg if is_pg else (share_pg * float(soft_factor_tm))
        if share <= 0.0:
            out[sname] = base_val
            continue

        admin_mult = max(0.0, get_admin(sid))
        copay = max(0.0, min(1.0, get_copay(sid)))

        priv_keep = base_val * (1.0 - share)
        social_val_admin = base_val * share * admin_mult
        co_payment = copay * social_val_admin
        subsidy    = (1.0 - copay) * social_val_admin

        out[sname] = max(0.0, priv_keep + co_payment)
        G_socialized_t += max(0.0, subsidy)

    return out, float(G_socialized_t)



def update_firm_budget_vectorized(firm_list, t, params):
    """
    Version vectorisée robuste de la mise à jour du budget des firmes.
    Compatible même si certaines clés n'existent pas encore.
    """

    budgets_last = np.array([f.get("Budget", [0])[-1] for f in firm_list])
    profits_t = np.array([f.get("Profits", [0])[t] if len(f.get("Profits", [])) > t else 0 for f in firm_list])
    rd_budget_last = np.array([f.get("RD_Budget", [0])[-1] for f in firm_list])
    switching_periods = np.array([f.get("Switching_Period", -1) for f in firm_list])
    switching_costs = np.array([f.get("Switching_Costs", 0) for f in firm_list])
    portfolios = np.array([f.get("Portfolio", [0])[-1] if isinstance(f.get("Portfolio"), list) else f.get("Portfolio", 0) for f in firm_list])
    new_firms = np.array([f.get("NewFirm", 0) for f in firm_list])
    rd_watch_last = np.array([f.get("RD_Watch", [0])[-1] for f in firm_list])

    # Cas nouveaux entrants : budget = budget moyen
    avg_budget = np.mean(budgets_last) if len(budgets_last) > 0 else 0
    new_budget = np.where(new_firms == 1, avg_budget, budgets_last + profits_t - rd_budget_last)

    # Ajouter switching costs
    new_budget -= np.where(switching_periods == t, switching_costs, 0)

    # Faillites si budget <= 0
    dead = (new_budget <= 0).astype(int)
    new_budget = np.where(dead == 1, 0, new_budget)

    # Réglementation (si produit 1 interdit et Portfolio=0)
    regulation = params.get("Regulation", 0)
    if regulation == 1:
        dead = np.where(portfolios == 0, 1, dead)

    # Vérification veille technologique
    sunset_date = params.get("Sunset_Date", 10)
    revision_period = params.get("Revision_Period", 5)
    revision_update = params.get("Revision_Period_Update", 0)
    if t == (sunset_date + revision_period - revision_update) and regulation == 0:
        avg_watch = np.mean(rd_watch_last) if len(rd_watch_last) > 0 else 0
        alpha_watch = params.get("Alpha_Regu_Watch", 1)
        dead = np.where((portfolios == 0) & (rd_watch_last < avg_watch * alpha_watch), 1, dead)

    # 🔒 Sécurisation : dead doit être un tableau 1D d'entiers
    dead = np.array(dead).astype(int).reshape(-1)

    # Réécrire dans firm_list
    for i, firm in enumerate(firm_list):
        firm.setdefault("Budget", []).append(float(new_budget[i]))
        firm.setdefault("Dead", []).append(int(dead[i]))


def update_innovation_and_adoption(firm_list, household_list, t, paramsinnov):
    """
    Met à jour parts de marché, adoption de la techno 2, R&D, innovation et régulation.
    Toutes les variables dynamiques sont stockées avec .append() pour conserver l’historique.
    """

    # === 1. Parts de marché ===
    total_sales = sum(f["Sales"][t] if len(f["Sales"]) > t else 0 for f in firm_list)
    for firm in firm_list:
        sales_t = firm["Sales"][t] if len(firm["Sales"]) > t else 0
        firm["MarketShare"] = sales_t / total_sales if total_sales > 0 else 0

    # === 2. Adoption Produit 2 ===
    for firm in firm_list:
        if firm["Dead"][-1] == 1:
            continue  # une firme morte ne peut plus adopter

        if "K" not in firm:
            firm["K"] = [0.1] * (t + 1)  # valeur initiale
        elif len(firm["K"]) <= t:
            firm["K"].append(firm["K"][-1])  # prolonge la série

        # Accès au portefeuille actuel
        portfolio = firm["Portfolio"][-1]

        if portfolio == 0:  # pas encore adopté P2
            adoption_index = firm["MarketShare"] * (
                1 + paramsinnov["Alpha_Regu"] * t / (paramsinnov["Sunset_Date"] + paramsinnov["Revision_Period"])
            )
            K_value = safe_get(firm["K"], t, 0)
            if adoption_index > firm["AdoptionThreshold_MS"] and K_value > firm["AdoptionThreshold_K"]:
                budget_t = firm["Budget"][t]
                if budget_t >= firm["Switching_Costs"]:
                    firm["Portfolio"].append(1)  # passage en P1+P2
                    firm["Switching_Period"] = t + 1
                    # Initialisation techno 2 à l’adoption
                    firm["X_P2"].append(paramsinnov["Product2_Init_X"])
                    firm["Eff_P2"].append(paramsinnov["Product2_Init_Eff"])
                    firm["Tox_P2"].append(paramsinnov["Product2_Init_Tox"])
                    firm["Bio_P2"].append(paramsinnov["Product2_Init_Bio"])
                else:
                    # pas assez de budget → pas d’adoption
                    firm["X_P2"].append(firm["X_P2"][-1])
                    firm["Eff_P2"].append(firm["Eff_P2"][-1])
                    firm["Tox_P2"].append(firm["Tox_P2"][-1])
                    firm["Bio_P2"].append(firm["Bio_P2"][-1])
            else:
                # pas encore adopté → on reporte les valeurs existantes
                firm["X_P2"].append(firm["X_P2"][-1])
                firm["Eff_P2"].append(firm["Eff_P2"][-1])
                firm["Tox_P2"].append(firm["Tox_P2"][-1])
                firm["Bio_P2"].append(firm["Bio_P2"][-1])

    # === 3. Allocation du budget R&D ===
    for firm in firm_list:
        if firm["Dead"][-1] == 1:
            firm["RD1"].append(0)
            firm["RD2"].append(0)
            firm["RD_Watch"].append(0)
            continue

        budget_t = firm["Budget"][t]
        rd_total = paramsinnov["RDshare"] * budget_t

        portfolio = firm["Portfolio"][-1]

        if portfolio == 0:  # seulement P1
            rd1 = firm["RDSplit"] * rd_total
            rd2 = 0
            rd_watch = (1 - firm["RDSplit"]) * rd_total
        elif portfolio == 1:  # P1 et P2
            rd1 = firm["RDSplit"] * rd_total
            rd2 = (1 - firm["RDSplit"]) * rd_total
            rd_watch = 0
        else:  # seulement P2
            rd1 = 0
            rd2 = rd_total
            rd_watch = 0

        firm["RD1"].append(rd1)
        firm["RD2"].append(rd2)
        firm["RD_Watch"].append(rd_watch)

    # === 4. Innovation stochastique ===
    for firm in firm_list:
        if firm["Dead"][-1] == 1:
            continue

        portfolio = firm["Portfolio"][-1]

        # Produit 1
        if portfolio in [0, 1]:
            prob_innov = 1 - np.exp(-paramsinnov["scale"] * firm["RD1"][-1])
            # X
            if np.random.rand() < prob_innov:
                firm["X_P1"].append(min(firm["X_P1"][-1] + paramsinnov["Alpha_X"], paramsinnov["Xmax_Prod1"]))
            else:
                firm["X_P1"].append(firm["X_P1"][-1])
            # Eff
            if np.random.rand() < prob_innov:
                firm["Eff_P1"].append(min(firm["Eff_P1"][-1] + paramsinnov["Alpha_Eff"], paramsinnov["Effmax_Prod1"]))
            else:
                firm["Eff_P1"].append(firm["Eff_P1"][-1])
            # Tox
            if np.random.rand() < prob_innov:
                firm["Tox_P1"].append(max(firm["Tox_P1"][-1] - paramsinnov["Alpha_Tox"], paramsinnov["Toxmin_Prod1"]))
            else:
                firm["Tox_P1"].append(firm["Tox_P1"][-1])
            # Bio
            if np.random.rand() < prob_innov:
                firm["Bio_P1"].append(max(firm["Bio_P1"][-1] - paramsinnov["Alpha_Bio"], paramsinnov["Biomin_Prod1"]))
            else:
                firm["Bio_P1"].append(firm["Bio_P1"][-1])

        # Produit 2
        if portfolio in [1, 2]:
            prob_innov = 1 - np.exp(-paramsinnov["scale"] * firm["RD2"][-1])
            # X
            if np.random.rand() < prob_innov:
                firm["X_P2"].append(min(firm["X_P2"][-1] + paramsinnov["Alpha_X"], paramsinnov["Xmax_Prod2"]))
            else:
                firm["X_P2"].append(firm["X_P2"][-1])
            # Eff
            if np.random.rand() < prob_innov:
                firm["Eff_P2"].append(min(firm["Eff_P2"][-1] + paramsinnov["Alpha_Eff"], paramsinnov["Effmax_Prod2"]))
            else:
                firm["Eff_P2"].append(firm["Eff_P2"][-1])
            # Tox
            if np.random.rand() < prob_innov:
                firm["Tox_P2"].append(max(firm["Tox_P2"][-1] - paramsinnov["Alpha_Tox"], paramsinnov["Toxmin_Prod2"]))
            else:
                firm["Tox_P2"].append(firm["Tox_P2"][-1])
            # Bio
            if np.random.rand() < prob_innov:
                firm["Bio_P2"].append(max(firm["Bio_P2"][-1] - paramsinnov["Alpha_Bio"], paramsinnov["Biomin_Prod2"]))
            else:
                firm["Bio_P2"].append(firm["Bio_P2"][-1])

    # === 5. Régulation (simplifiée) ===
    if t == paramsinnov["Sunset_Date"] + paramsinnov["Revision_Period"]:
        eff_vals = [f["Eff_P2"][-1] for f in firm_list if f["Portfolio"][-1] > 0]
        x_vals = [f["X_P2"][-1] for f in firm_list if f["Portfolio"][-1] > 0]
        if eff_vals and x_vals:
            avg_eff_p2 = np.mean(eff_vals)
            avg_x_p2 = np.mean(x_vals)
            if avg_eff_p2 >= paramsinnov["Target_Eff"] and avg_x_p2 >= paramsinnov["Target_X"]:
                for firm in firm_list:
                    if firm["Portfolio"][-1] == 0:
                        firm["Dead"].append(1)
                    else:
                        firm["Dead"].append(0)
            else:
                for firm in firm_list:
                    firm["Dead"].append(firm["Dead"][-1])
        else:
            for firm in firm_list:
                firm["Dead"].append(firm["Dead"][-1])
    else:
        for firm in firm_list:
            firm["Dead"].append(firm["Dead"][-1])

    return firm_list


# Liste pour stocker tous les résultats
gini_records = []
all_results = []



# Simulation
for sim in range(num_simulations):
    max_brown_credit = 1.0  # 100% de capacité initialement
    scenario_names = ["carbon_tax_only", "transition_mix", "post_growth"]
    scenario_name = scenario_names[sim % len(scenario_names)]
    scenario = sim % len(scenario_names) + 1  # 1: carbon_tax_only, 2: transition_mix, 3: post_growth
    # --- Réinitialisation des variables de socialisation pour cette simulation ---
    for s in range(1, 7):
        K_shared[s] = [0.0] * num_periods
        I_pub_shared_by_sector[s] = [0.0] * num_periods

    for tt in range(num_periods):
        G_shared_list[tt] = 0.0
        I_pub_shared_list[tt] = 0.0
        caproom_left[tt] = float("inf")


    # Initialisation des politiques
    carbon_tax_active = False
    transition_policy_active = False
    post_growth_policy_active = False

    if scenario_name in ["carbon_tax_only", "transition_mix"]:
        carbon_tax_active = True
    elif scenario_name == "transition_mix":
        transition_policy_active = True
    elif scenario_name == "post_growth":
        post_growth_policy_active = True

    firm_list = []
    for i in range(num_firms):
        firm = {
            "FirmID": i,
            "ProductivityFactor": random.uniform(0.0, 0.3),
            "IdBank": random.randint(0, num_banks - 1),
            "BrownCapital": [0.0],
            "GreenCapital": [0.0],
            "BrownInvestment": [],
            "GreenInvestment": [],
            "BrownInvDes": [],
            "GreenInvDes": [],
            "BrownLoansDem": [],
            "GreenLoansDem": [],
            "BrownLoansVar": [],
            "GreenLoansVar": [],
            "Profits": [0.0],
            "TotalDebt": [0.0],
            "LoanInterestRate": [],
            "GreenLoanInterestRate": [],
            "AnimalSpirits_B": [],
            "AnimalSpirits_V": [],
            "gamma1": [],
            "gamma2": [],
            "gamma3": [],
            "gamma4": [],
            "GrKdB": [],
            "GrKdV": [],
            "ratiocashflow": [],
            "LeverageFirm": [],
            "FullCapacityProduction": [],
            "BrownLoans": [0.0],
            "GreenLoans": [0.0],
            "CreditConstraintVar": [],
            "CreditConstraintInt": [],
            "TotalCapital": [0.0],
            "GreenCapitalRatio": [],
            "IdSector": sectors.index(sectors[i % len(sectors)]) + 1,
            "SelectedChampions": 1 if random.random() < 0.2 else 0,  # 20% des firmes sont champions
            "Production": [0]*num_periods,
            # === Variables d’innovation ===
            "Portfolio": [0],   # toujours une liste
            "AdoptionThreshold_MS": np.random.uniform(0.01, 0.05),
            "AdoptionThreshold_K": np.random.uniform(0.01, 0.05),
            "Switching_Costs": np.random.uniform(0, 0.1),
            "Switching_Period": 0,
            "RDSplit": np.random.rand(),
            "Products": [
                    {   # Produit 1
                        "IdProduct": 1,
                        "IsAdopted": [1],
                        "X": [0.3],
                        "Eff": [0.2],
                        "Tox": [0.5],
                        "Bio": [0.5],
                        "Price": [0.1],
                    },
                    {   # Produit 2 (pas encore adopté)
                        "IdProduct": 2,
                        "IsAdopted": [0],
                        "X": [0.1],
                        "Eff": [0.1],
                        "Tox": [0.8],
                        "Bio": [0.8],
                        "Price": [0.1],
                    },
                ],
            "Sales": [0],  # ventes totales de la firme (mis à jour à chaque période)
            "Dead": [0],  # statut de vie/mort de la firme (0 = active, 1 = sortie)
            "Budget": [0],
            "X": [0],       # performance (sera MAJ chaque période)
            "RD1": [0],
            "RD2": [0],
            "RD_Watch": [0],

            # Produit 1 (init techno)
            "X_P1": [paramsinnov["Product1_Init_X"]],
            "Eff_P1": [paramsinnov["Product1_Init_Eff"]],
            "Tox_P1": [paramsinnov["Product1_Init_Tox"]],
            "Bio_P1": [paramsinnov["Product1_Init_Bio"]],

            # Produit 2 (init techno)
            "X_P2": [paramsinnov["Product2_Init_X"]],
            "Eff_P2": [paramsinnov["Product2_Init_Eff"]],
            "Tox_P2": [paramsinnov["Product2_Init_Tox"]],
            "Bio_P2": [paramsinnov["Product2_Init_Bio"]],

        }
        if random.random() < 0.51:
            firm["Portfolio"] = [1]                  # portfolio hybride
            firm["Products"][1]["IsAdopted"] = [1]   # le produit 2 est déjà adopté
            # Produit 1 reste adopté (IsAdopted = [1]), donc bien brun + vert

        firm_list.append(firm)

    # === Init des champs liés au module "shared" (à placer juste après la création de firm_list) ===
    for f in firm_list:
        f.setdefault("AlphaShared", [])     # part de prod dédiée au shared (indicatif)
        f.setdefault("SharedRevenue", [])   # recette publique (contrats shared)


    household_list, firm_list = initialize_households_and_sales_from_struct(
        households_struct, firm_list, params, num_sectors=6,
        rural_prefs=rural_prefs, urban_prefs=urban_prefs
    )

    bank_list = []
    for b in range(num_banks):
        bank = {
            "BankID": b,
            "TargetLeverage": [],
            "BanksAnimalSpirits": [],
            "Failures": [random.randint(0, 3)],
            "RiskAppetite": [],
            "GreenLoansBank": [],
            "BrownLoansBank": [],
            "GreenAssetRatio": [],
            "BankLoans": [],
            "MarketShareBank": [],
        }
        bank_list.append(bank)

    # ========================================================
    #   CALIBRATION EXACTE DU CAPITAL (BRUN + VERT)
    #   À PARTIR DES DONNÉES FIXED ASSETS DE L'EXCEL
    # ========================================================

    # Capital fixe total par secteur (en millions EUR)
    fixed_assets_sector = {
        "Agriculture": 1785.78087,
        "Energy":      2483.52562,
        "Housing":     1616.43662,
        "Transport":   4509.23822,
        "Industry":    6911.55628,
        "Technology":  2457.81971,
    }

    # Part verte initiale : très faible dans la réalité (5 % par défaut)
    INITIAL_GREEN_SHARE = 0.05

    # Classification des firmes par secteur (via IdSector → nom de secteur)
    firms_by_sector = {s: [] for s in sectors}
    for f in firm_list:
        sec_name = sectors[f["IdSector"] - 1]   # IdSector = 1..6
        firms_by_sector[sec_name].append(f)

    # Répartition du capital vert / brun à partir de l’Excel
    for sec, cap_total in fixed_assets_sector.items():
        fs = firms_by_sector.get(sec, [])
        if len(fs) == 0:
            continue

        # Capital vert et brun sectoriels
        cap_green = cap_total * INITIAL_GREEN_SHARE
        cap_brown = cap_total * (1 - INITIAL_GREEN_SHARE)

        # Répartition uniforme entre firmes du secteur
        green_per_firm = cap_green / len(fs)
        brown_per_firm = cap_brown / len(fs)

        for f in fs:
            f["GreenCapital"][0] = green_per_firm
            f["BrownCapital"][0] = brown_per_firm

    # ========================================================
    #   CALIBRATION EXACTE DES PRÊTS BRUNS / PRÊTS VERTS
    #   À PARTIR DES DONNÉES EXCEL (ONGLET FINANCE)
    # ========================================================

    # ---- 1. Green loans sectoriels (Excel, en milliards) ----
    green_loans_sector = {
        "Agriculture": 22.8,
        "Energy":      99.5,
        "Housing":     60.0,
        "Transport":   80.5,
        "Industry":    86.3,
        "Technology":  42.7,
    }

    # ---- 2. Total green loans macro ----
    total_green_loans = sum(green_loans_sector.values())   # ≈ 454 milliards

    # ---- 3. Part des green loans dans le loan book total (Excel = 4 %) ----
    GREEN_SHARE_MACRO = 0.04

    # Donc : BrownLoansTotal = GreenLoansTotal * 24
    total_brown_loans = total_green_loans * (1 - GREEN_SHARE_MACRO) / GREEN_SHARE_MACRO

    # ---- 4. Répartition des green loans par secteur ----
    for sec, amount_sec in green_loans_sector.items():
        fs = firms_by_sector.get(sec, [])
        if len(fs) > 0:
            per_firm = amount_sec / len(fs)
            for f in fs:
                f["GreenLoans"][0] = per_firm
        else:
            # si jamais un secteur n'a aucune firme (normalement impossible)
            pass

    # ---- 5. Répartition des brown loans proportionnellement au capital brun ----
    # Très important : cohérence économique + cohérence SFC
    brown_cap_total = sum(f["BrownCapital"][0] for f in firm_list)

    for f in firm_list:
        share = f["BrownCapital"][0] / brown_cap_total if brown_cap_total > 0 else 0.0
        f["BrownLoans"][0] = share * total_brown_loans
        f["TotalDebt"][0] = f["GreenLoans"][0] + f["BrownLoans"][0]

    # ---- 6. Agrégation bancaire des prêts bruns/verts ----
    for bank in bank_list:
        g_sum = 0.0
        b_sum = 0.0

        for f in firm_list:
            if f["IdBank"] == bank["BankID"]:
                g_sum += f["GreenLoans"][0]
                b_sum += f["BrownLoans"][0]

        bank["GreenLoansBank"] = [g_sum]
        bank["BrownLoansBank"] = [b_sum]


    central_bank_rate = [0.02] * num_periods
    bandwagon_effect = []
    TransitionActive = []

    base_wage = 0.2  # salaire de base de ton modèle

    # Salaire minimum calibré comme 2/3 du salaire de base
    MIN_WAGE_RATIO = 2.0 / 3.0
    minimum_wage = [base_wage * MIN_WAGE_RATIO]

    # Taux de remplacement du chômage (cohérent avec les ordres de grandeur UE)
    UB_REPLACEMENT = 0.60

    # Niveau de référence d’allocation chômage (peu utilisé après patch, mais on le garde pour compatibilité)
    unemployment_benefit = [minimum_wage[0] * UB_REPLACEMENT],

    green_premium_low = 0.08      # Excel : 8 % low-skilled
    green_premium_high = 0.02     # Excel : 2 % high-skilled
    skill_premium = 0.20          # provisoire : 20 % de prime de compétence
    firms_by_bank = {b: [] for b in range(num_banks)}
    for firm in firm_list:
        firms_by_bank[firm["IdBank"]].append(firm)

    # ==========================
    # PARAMÈTRES BANQUES (calibration Excel)
    # ==========================

    # Part des firmes déclarant la finance comme obstacle majeur (Excel "Behavioural parameters Firms", ligne 19)
    FINANCE_OBSTACLE_SHARE = 0.18

    # Levier cible toléré par les banques pour les firmes : dette / capital ≈ 2
    # (ordre de grandeur stylisé faute de vraie donnée explicite dans l’Excel)
    BANK_TARGET_LEVERAGE = 2.0

    # Intensité de réaction des banques dans la contrainte de crédit (bank "animal spirits")
    BANK_SPIRITS = 1.0,


    #seed_initial_shared_capital_from_private0(firm_list, K_shared, seed_ratio_by_sector, scenario_name, PostGrowthActive)

    # === Flags politiques pré-dimensionnés (endogènes par élections) ===
    TransitionActive = [0] * num_periods
    CarbonTaxActive  = [0] * num_periods
    PostGrowthActive = [0] * num_periods

    ELECTIONS   = [5, 10, 15, 20]   # périodes de vote
    MANDATE_LEN = 5                 # durée du mandat (périodes)

    def majority_wants_policy(t):
        """True si majorité pour à t, False si contre, None si votes incomplets."""
        # Tous les ménages doivent avoir VoteDecision[t]
        if not all(len(h.get("VoteDecision", [])) > t for h in household_list):
            return None
        nb_pro = sum(1 for h in household_list if h["VoteDecision"][t] == 1)
        return (nb_pro > (len(household_list) // 2))

    seeded_shared = False

    for t in range(num_periods):
        # --- Semis du capital public partagé à partir du capital privé à t=0 ---
        seed_initial_shared_capital_from_private0(
            firm_list,
            K_shared,
            seed_ratio_by_sector,
            scenario_name,
            PostGrowthActive
        )

# UNSURE ABOUT THAT BELOW
#        update_policy_activation(t, household_list, CarbonTaxActive, scenario == 1)
#        update_policy_activation(t, household_list, TransitionActive, scenario == 2)
#        update_policy_activation(t, household_list, PostGrowthActive, scenario == 3)
        # Réduction progressive de la capacité de prêts verts
        def _consecutive_active(active_list, t):
            k, i = 0, t
            while i >= 0 and active_list[i] == 1:
                k += 1
                i -= 1
            return k

        if scenario == "post_growth" and t > 0 and PostGrowthActive is not None and PostGrowthActive[t] == 1:
            # 1) Mémoriser le plafond initial la première fois qu'on entre en post-growth
            if POLICY_STATE["brown_credit_initial_cap"] is None:
                POLICY_STATE["brown_credit_initial_cap"] = float(max_brown_credit)

            # 2) Paramétrage de sévérité (ajuste à ta convenance)
            factor0 = 0.60          # contraction exponentielle par période active (0.6 = -40%/période active)
            hard_floor_ratio = 0.05 # 5% du plafond initial maximum toléré
            extinction_after = 6    # après 6 périodes actives: extinction totale du crédit brun

            # 3) Compter le nombre de périodes post-growth consécutives jusqu'à t
            streak = _consecutive_active(PostGrowthActive, t)

            # 4) Calculer un plafond décrivant une rampe dure vers 0
            initial_cap = POLICY_STATE["brown_credit_initial_cap"]
            # plafond exponentiel qui descend vite, mais ne passe pas sous le "hard floor"
            exp_cap = initial_cap * (factor0 ** streak)
            floor_cap = initial_cap * hard_floor_ratio
            cap_t = max(floor_cap, exp_cap)

            # 5) Extinction totale après N périodes actives
            if streak >= extinction_after:
                cap_t = 0.0

            # 6) Appliquer le nouveau plafond au crédit brun disponible
            max_brown_credit = max(0.0, min(float(max_brown_credit), cap_t))

        if t == 0:
            brown_loan_cap.append(1.0)  # Autorisation initiale à 100%
        else:
            if PostGrowthActive[t] == 1:
                brown_loan_cap.append(max(0, brown_loan_cap[t - 1] - 0.80))
            else:
                brown_loan_cap.append(brown_loan_cap[t - 1])
        
        if t < 2:
            bandwagon_effect.append(0.0)
        else:
            n_0to1 = 0
            n_1to0 = 0

            for household in household_list:
                # --- Sécurisation du vote deux périodes avant ---
                if "VoteDecision" not in household:
                    household["VoteDecision"] = [0]

                # Si la liste n’a pas encore deux éléments, valeur par défaut
                if len(household["VoteDecision"]) <= t - 2 or t - 2 < 0:
                    vote_t2 = 0  # 0 = abstention / neutre
                else:
                    vote_t2 = household["VoteDecision"][t - 2]

                if "VoteDecision" not in household or len(household["VoteDecision"]) <= t - 1:
                    vote_t1 = 0  # ou une valeur par défaut (0 = abstention / neutre)
                else:
                    vote_t1 = household["VoteDecision"][t - 1]
                if "VoteDecision" not in household:
                    household["VoteDecision"] = [0]  # 0 = abstention par défaut

                # Si la liste est vide ou trop courte, on prend 0 par défaut
                if len(household["VoteDecision"]) == 0 or t - 1 < 0:
                    vote_t1 = 0
                elif len(household["VoteDecision"]) <= t - 1:
                    vote_t1 = household["VoteDecision"][-1]  # on prend la dernière connue
                else:
                    vote_t1 = household["VoteDecision"][t - 1]
                if vote_t2 == 0 and vote_t1 == 1:
                    n_0to1 += 1
                if vote_t2 == 1 and vote_t1 == 0:
                    n_1to0 += 1

            net_shift = n_0to1 - n_1to0
            bandwagon = net_shift / num_households
            bandwagon_effect.append(bandwagon)


        if t == 0:
            minimum_wage = [0.6]
        else:
            prev_wage = minimum_wage[t - 1]
            increase_rate = 0.02 if TransitionActive[t - 1] == 1 else 0.0
            new_wage = prev_wage * (1 + increase_rate)
            minimum_wage.append(new_wage)

        if t == 0:
            unemployment_benefit = [2]
        else:
            prev_ub = unemployment_benefit[t - 1]
            ub_increase_rate = 0.02 if TransitionActive[t - 1] == 1 else 0.02
            new_ub = prev_ub * (1 + ub_increase_rate)
            unemployment_benefit.append(new_ub)


# === NOUVEAU BLOC : Production dynamique et demande d'emploi fluctuante ===

        for firm in firm_list:
                    # --- 1. Production dépendante du capital total ---
            green_capital = firm.get("GreenCapital", [0])[t - 1] if t > 0 else 0
            brown_capital = firm.get("BrownCapital", [0])[t - 1] if t > 0 else 0
            total_capital = green_capital + brown_capital

            # Calcul de l'intensité énergétique microéconomique
            energy_intensity = compute_firm_energy_intensity(firm, t, params["epsilon_V"], params["epsilon_B"])
    
            # Production ou consommation adressée à la firme
            production = firm.get("Production", [0]*num_periods)[t]
    
            # Calcul des besoins énergétiques (à utiliser ensuite dans update_biophys_vars)
            firm["EnergyNeeded"] = energy_intensity * production


    # Productivité endogène croissante avec le capital vert
            base_productivity = 30
            gamma = 20
            productivity = base_productivity + gamma * (green_capital / (green_capital + brown_capital + 1e-6))

    # Production maximale avec bruit aléatoire (efficacité variable)
            shock = random.uniform(0.85, 1.15)
            production = total_capital * productivity * shock

    # Mémoriser cette production
            if "Production" not in firm:
                firm["Production"] = [0] * num_periods
            firm["Production"][t] = production

    # --- 2. Recalcul de la demande d'emplois ---
            green_ratio = firm.get("GreenCapitalRatio", [0])[t - 1] if t > 0 else 0.5
            brown_ratio = 1 - green_ratio

            base_productivity_demand = 60
        
            green_jobs_total = (production / base_productivity_demand) / green_ratio if green_ratio > 0 else 0
            brown_jobs_total = (production / base_productivity_demand) / brown_ratio if brown_ratio > 0 else 0

    # Fluctuation réaliste de l'organisation du travail
            green_high = int(green_jobs_total * random.uniform(0.4, 0.6))
            green_low = int(green_jobs_total - green_high)
            brown_high = int(brown_jobs_total * random.uniform(0.4, 0.6))
            brown_low = int(brown_jobs_total - brown_high)

            firm_id = firm["FirmID"]

            current_green_high = sum(1 for h in household_list if h["IdEmployer"] == firm_id and h["Status"] == 5)
            current_green_low = sum(1 for h in household_list if h["IdEmployer"] == firm_id and h["Status"] == 2)
            current_brown_high = sum(1 for h in household_list if h["IdEmployer"] == firm_id and h["Status"] == 4)
            current_brown_low = sum(1 for h in household_list if h["IdEmployer"] == firm_id and h["Status"] == 1)

            firm["GreenHighSkilledDemand"] = green_high - current_green_high
            firm["GreenLowSkilledDemand"] = green_low - current_green_low
            firm["BrownHighSkilledDemand"] = brown_high - current_brown_high
            firm["BrownLowSkilledDemand"] = brown_low - current_brown_low

        """ # === BOUCLE DE MISE À JOUR DU STATUT D'EMPLOI SELON DEMANDE NETTE ===
        # === BOUCLE DE MISE À JOUR DU STATUT D'EMPLOI SELON DEMANDE NETTE ===

        # === BOUCLE DE MISE À JOUR DU STATUT D'EMPLOI SELON DEMANDE NETTE (avec distinction chômage vs emploi garanti) ===

        for h, household in enumerate(household_list):
            status = household["Status"]
            skill = household["SkillStatus"]
            employer_id = household["IdEmployer"]
            reskilling_counter = household.get("ReskillingCounter", 0)
           
            brown_retention = 0.8
            green_retention = 0.85
            brown_qualified_retention = 0.9
            green_qualified_retention = 0.92

                # --- 1. Formation (Status = 6)
            if status == 6:
                    reskilling_counter += 1
                    if reskilling_counter >= 2:
                            household["Status"] = 0  # retour au chômage classique
                            household["IdEmployer"] = -1
                            reskilling_counter = 0
                    household["ReskillingCounter"] = reskilling_counter
                    continue

                # --- 2. Emploi garanti si au chômage et transition active, avec probabilité
            if status == 0 and TransitionActive[t] == 1 and random.random() < 0.3:
                    household["Status"] = 3  # emploi garanti
                    household["IdEmployer"] = -1
                    household["Income"].append(minimum_wage[t])
                    continue

                # --- 3. Licenciements aléatoires (5% turnover) ou par demande nette
            if status in [1, 2, 4, 5] and employer_id != -1:
                    firm = firm_list[employer_id]
                    fired = False

                    if random.random() < 0.05:
                            fired = True
                    elif status == 1 and firm["BrownLowSkilledDemand"] < 0 and random.random() > 0.8:
                            firm["BrownLowSkilledDemand"] += 1
                            fired = True
                    elif status == 2 and firm["GreenLowSkilledDemand"] < 0 and random.random() > 0.85:
                            firm["GreenLowSkilledDemand"] += 1
                            fired = True
                    elif status == 4 and firm["BrownHighSkilledDemand"] < 0 and random.random() > 0.9:
                            firm["BrownHighSkilledDemand"] += 1
                            fired = True
                    elif status == 5 and firm["GreenHighSkilledDemand"] < 0 and random.random() > 0.92:
                            firm["GreenHighSkilledDemand"] += 1
                            fired = True

                    if fired:
                            household["Status"] = 0  # retour au chômage
                            household["IdEmployer"] = -1
                            continue

                # --- 4. Embauche si au chômage (Status = 0) et demande positive
            if household["Status"] == 0:
                    random.shuffle(firm_list)
                    for firm in firm_list:
                            fid = firm["FirmID"]

                            if skill == 0:
                                    if firm["BrownLowSkilledDemand"] > 0:
                                            household["Status"] = 1
                                            household["IdEmployer"] = fid
                                            firm["BrownLowSkilledDemand"] -= 1
                                            break
                                    elif firm["GreenLowSkilledDemand"] > 0:
                                            household["Status"] = 2
                                            household["IdEmployer"] = fid
                                            firm["GreenLowSkilledDemand"] -= 1
                                            break

                            elif skill == 1:
                                    if firm["BrownHighSkilledDemand"] > 0:
                                            household["Status"] = 4
                                            household["IdEmployer"] = fid
                                            firm["BrownHighSkilledDemand"] -= 1
                                            break
                                    elif firm["GreenHighSkilledDemand"] > 0:
                                            household["Status"] = 5
                                            household["IdEmployer"] = fid
                                            firm["GreenHighSkilledDemand"] -= 1
                                            break
        """ 

        # === BOUCLE DE MISE À JOUR DU STATUT D'EMPLOI SELON DEMANDE NETTE (version calibrée) ===

        for h, household in enumerate(household_list):
            status = household["Status"]
            skill = household["SkillStatus"]
            employer_id = household["IdEmployer"]
            reskilling_counter = household.get("ReskillingCounter", 0)

            # --- 1. Formation (Status = 6)
            if status == 6:
                reskilling_counter += 1
                if reskilling_counter >= 2:
                    household["Status"] = 0  # retour chômage classique
                    household["IdEmployer"] = -1
                    reskilling_counter = 0
                household["ReskillingCounter"] = reskilling_counter
                continue

            # --- 2. Emploi garanti (JG) si transition active
            if status == 0 and TransitionActive[t] == 1 and random.random() < 0.3:
                household["Status"] = 3  # emploi garanti
                household["IdEmployer"] = -1
                # Le revenu sera calculé plus bas dans le bloc "Income"
                continue

            # --- 3. Licenciements (probabilité réduite)
            if status in [1, 2, 4, 5] and employer_id != -1:
                firm = firm_list[employer_id]
                fired = False

                # turnover réduit
                if random.random() < 0.02:
                    fired = True
                # licenciements conditionnels (moins sévères)
                elif status == 1 and firm["BrownLowSkilledDemand"] < 0 and random.random() > 0.9:
                    firm["BrownLowSkilledDemand"] += 1
                    fired = True
                elif status == 2 and firm["GreenLowSkilledDemand"] < 0 and random.random() > 0.9:
                    firm["GreenLowSkilledDemand"] += 1
                    fired = True
                elif status == 4 and firm["BrownHighSkilledDemand"] < 0 and random.random() > 0.93:
                    firm["BrownHighSkilledDemand"] += 1
                    fired = True
                elif status == 5 and firm["GreenHighSkilledDemand"] < 0 and random.random() > 0.95:
                    firm["GreenHighSkilledDemand"] += 1
                    fired = True

                if fired:
                    household["Status"] = 0  # retour chômage
                    household["IdEmployer"] = -1
                    continue

            # --- 4. Embauches si au chômage (Status=0) et demande positive
            if household["Status"] == 0:
                random.shuffle(firm_list)
                for firm in firm_list:
                    fid = firm["FirmID"]

                    if skill == 0:
                        if firm["BrownLowSkilledDemand"] > 0:
                            household["Status"] = 1
                            household["IdEmployer"] = fid
                            firm["BrownLowSkilledDemand"] -= 1
                            break
                        elif firm["GreenLowSkilledDemand"] > 0:
                            household["Status"] = 2
                            household["IdEmployer"] = fid
                            firm["GreenLowSkilledDemand"] -= 1
                            break

                    elif skill == 1:
                        if firm["BrownHighSkilledDemand"] > 0:
                            household["Status"] = 4
                            household["IdEmployer"] = fid
                            firm["BrownHighSkilledDemand"] -= 1
                            break
                        elif firm["GreenHighSkilledDemand"] > 0:
                            household["Status"] = 5
                            household["IdEmployer"] = fid
                            firm["GreenHighSkilledDemand"] -= 1
                            break


        # --- BOUCLE BANQUES (CALIBRÉE) ---
        for bank in bank_list:
            # On garde une petite hétérogénéité de levier cible entre banques,
            # mais fixe dans le temps. Les autres paramètres sont constants.

            if t == 0:
                # Levier cible par banque : autour de BANK_TARGET_LEVERAGE, +/-10 %
                tl = BANK_TARGET_LEVERAGE * (1 + np.random.uniform(-0.1, 0.1))
                bank["TargetLeverage"].append(tl)

                # Animal spirits des banques : intensité de réaction de la contrainte de crédit
                bank["BanksAnimalSpirits"].append(BANK_SPIRITS)

                # Appétit pour le risque : 1 - part des firmes pour qui la finance est un obstacle
                # (Excel "finance is a major obstacle" ≈ 0.18)
                bank["RiskAppetite"].append(1.0 - FINANCE_OBSTACLE_SHARE)

                # Part de marché bancaire : on part sur des parts égales, qui SOMMENT à 1
                bank["MarketShareBank"].append(1.0 / num_banks)

            else:
                # Paramètres constants dans le temps : on recopie la valeur de t=0
                bank["TargetLeverage"].append(bank["TargetLeverage"][0])
                bank["BanksAnimalSpirits"].append(bank["BanksAnimalSpirits"][0])
                bank["RiskAppetite"].append(bank["RiskAppetite"][0])
                bank["MarketShareBank"].append(bank["MarketShareBank"][0])

        # --- BOUCLE FIRMES ---
        for firm in firm_list:
            # Initialisation des constantes ou paramètres dynamiques
            for key, value in {
                "LoanInterestRate": random.uniform(0.01, 0.05),
                "GreenLoanInterestRate": random.uniform(0.01, 0.05),
                "AnimalSpirits_B": random.uniform(0, 0.02),
                "AnimalSpirits_V": random.uniform(0, 0.02),
                "gamma1": 0.01,
                "gamma2": 0.01,
                "gamma3": 0.01,
                "gamma4": 0.01
            }.items():
                if key not in firm:
                    firm[key] = []
                firm[key].append(value)

            if t == 0:
                # Initialisation sécurisée des listes pour t = 0
                for key in [
                    "BaseWage", "LowSkilledBrownWage", "LowSkilledGreenWage",
                    "HighSkilledBrownWage", "HighSkilledGreenWage",
                    "BrownInvestment", "GreenInvestment", "BrownInvDes", "GreenInvDes",
                    "BrownLoansDem", "GreenLoansDem", "BrownLoansVar", "GreenLoansVar",
                    "CreditConstraintVar", "CreditConstraintInt",
                    "TotalCapital", "GreenCapitalRatio", "ratiocashflow", "LeverageFirm",
                    "FullCapacityProduction", "Investment", "Loans",
                    "GreenInvestmentShare", "BrownInvestmentShare"
                ]:
                    firm[key] = [0]

                # Assurer que TotalDebt existe à t = 0
                if "TotalDebt" not in firm:
                    firm["TotalDebt"] = [0]

                # Initialisation des prêts (divisé en deux)
                firm["BrownLoans"] = [firm["TotalDebt"][0] / 2]
                firm["GreenLoans"] = [firm["TotalDebt"][0] / 2]

                # Initialisation dépendant du capital
                total_capital = firm["BrownCapital"][0] + firm["GreenCapital"][0]
                firm["TotalCapital"][0] = total_capital
                firm["GreenCapitalRatio"][0] = (
                    firm["GreenCapital"][0] / total_capital if total_capital > 0 else 0
                )
                firm["FullCapacityProduction"][0] = 0.8 * total_capital
            else:

                base = base_wage * firm.get("ProductivityFactor", 1.0) * np.prod([(1 + general_inflation[i]) for i in range(t + 1)])
                firm["BaseWage"].append(base)
                low_brown_wage  = base_wage
                low_green_wage  = base_wage * (1 + green_premium_low)
                high_brown_wage = base_wage * (1 + skill_premium)
                high_green_wage = high_brown_wage * (1 + green_premium_high)

                firm["LowSkilledBrownWage"].append(low_brown_wage)
                firm["LowSkilledGreenWage"].append(low_green_wage)
                firm["HighSkilledBrownWage"].append(high_brown_wage)
                firm["HighSkilledGreenWage"].append(high_green_wage)
          
                # Fallback sécurisé si LoanInterestRate ou GreenLoanInterestRate sont absents
                if len(firm["LoanInterestRate"]) <= t - 1:
                    firm["LoanInterestRate"].append(0.02)
                if len(firm["GreenLoanInterestRate"]) <= t - 1:
                    firm["GreenLoanInterestRate"].append(0.015)

                                # Ratios financiers nécessaires
            prev_capital = firm["TotalCapital"][t - 1] if len(firm["TotalCapital"]) > t - 1 else 1
            prev_profits = firm["Profits"][0]  # profits constants dans cette version
            prev_debt = firm["TotalDebt"][t - 1] if len(firm["TotalDebt"]) > t - 1 else 100

            firm["ratiocashflow"].append(prev_profits / prev_capital if prev_capital > 0 else 0)
            firm["LeverageFirm"].append(prev_debt / prev_capital if prev_capital > 0 else 0)
            firm["FullCapacityProduction"].append(0.8 * prev_capital)

            # --- Taxe carbone éventuelle
            carbon_cost = 0
            if carbon_tax_active:
                carbon_tax_rate = 0.05
                carbon_cost = carbon_tax_rate * firm["BrownCapital"][t - 1]
            firm.setdefault("CarbonCost", []).append(carbon_cost)

            # === Bloc investissement corrigé ===
            # === Bloc investissement avec symétrie Brown / Green ===

            # --- Croissance désirée
            growth_b = max(0,
                firm["AnimalSpirits_B"][t]
                + firm["gamma1"][t] * firm["ratiocashflow"][t]
                - firm["gamma2"][t] * firm["LeverageFirm"][t]
                + firm["gamma3"][t] * 0.7
                - firm["gamma4"][t] * firm["LoanInterestRate"][t - 1]
                - 0.001 * carbon_cost
            )

            growth_v = max(0,
                firm["AnimalSpirits_V"][t]
                + firm["gamma1"][t] * firm["ratiocashflow"][t]
                - firm["gamma2"][t] * firm["LeverageFirm"][t]
                + firm["gamma3"][t] * 0.7
                - firm["gamma4"][t] * firm["GreenLoanInterestRate"][t - 1]
            )

            firm["GrKdB"].append(growth_b)
            firm["GrKdV"].append(growth_v)

            # --- Investissement désiré brut
            # === Bloc investissement corrigé ===

            # --- Demandes désirées (issues de growth_b / growth_v déjà calculés)
            brown_inv_des = max(0, growth_b * firm["BrownCapital"][t - 1])
            green_inv_des = max(0, growth_v * firm["GreenCapital"][t - 1])

            # On stocke les investissements désirés
            firm["BrownInvDes"].append(brown_inv_des)
            firm["GreenInvDes"].append(green_inv_des)

            # Les demandes de prêts sont toujours égales aux investissements désirés
            firm["BrownLoansDem"].append(brown_inv_des)
            firm["GreenLoansDem"].append(green_inv_des)

            # --- Contraintes bancaires
            bank = bank_list[firm["IdBank"]]
            target_leverage = bank["TargetLeverage"][t]
            bank_spirits = bank["BanksAnimalSpirits"][t]

            leverage_gap = firm["LeverageFirm"][t - 1] - target_leverage
            # --- Sécurisation : convertir en float si tuple
            if isinstance(bank_spirits, tuple):
                bank_spirits = bank_spirits[0]
            if isinstance(target_leverage, tuple):
                target_leverage = target_leverage[0]
            cc_raw = 1 / (1 + np.exp(-bank_spirits * leverage_gap))

            prev_constraint = firm["CreditConstraintVar"][t - 1] if t > 0 else 0
            cc_final = 0 * prev_constraint + 0 * cc_raw   # ici contraintes neutralisées

            firm["CreditConstraintVar"].append(cc_final)
            firm["CreditConstraintInt"].append(cc_final)

            # --- Taux d’intérêt
            cb_rate = central_bank_rate[t]
            market_share = bank["MarketShareBank"][t]
            markup = 0.02 * market_share

            firm["LoanInterestRate"][t] = cb_rate + markup + cc_final * 0.02
            firm["GreenLoanInterestRate"][t] = firm["LoanInterestRate"][t] - 0.01

            # --- Application des contraintes
            brown_cap = brown_loan_cap[t]
            allowed_brown = brown_cap * brown_inv_des

            brown_loans_granted = min(brown_inv_des * (1 - cc_final), allowed_brown)
            green_loans_granted = min(green_inv_des * (1 - cc_final), green_inv_des)

            firm["BrownLoansVar"].append(brown_loans_granted)
            firm["GreenLoansVar"].append(green_loans_granted)

            # --- Investissement final
            brown_investment = brown_loans_granted
            if scenario == "post_growth":
                brown_investment *= max_brown_credit

            green_investment = green_loans_granted

            firm["BrownInvestment"].append(brown_investment)
            firm["GreenInvestment"].append(green_investment)

            # --- Mise à jour capital et dettes
            prev_brown_cap = firm["BrownCapital"][t - 1]

            if PostGrowthActive[t] == 1:
                updated_brown_cap = max(0, prev_brown_cap * 0.80 + brown_investment)
            else:
                updated_brown_cap = prev_brown_cap + brown_investment

            firm["BrownCapital"].append((updated_brown_cap) * 1)
            firm["GreenCapital"].append((firm["GreenCapital"][t - 1]) * 1 + green_investment)

            firm["BrownLoans"].append(firm["BrownLoans"][t - 1] + brown_loans_granted)
            firm["GreenLoans"].append(firm["GreenLoans"][t - 1] + green_loans_granted)

            total_capital = firm["BrownCapital"][t] + firm["GreenCapital"][t]
            firm["TotalCapital"].append(total_capital)
            firm["TotalDebt"].append(firm["BrownLoans"][t] + firm["GreenLoans"][t])
            firm["GreenCapitalRatio"].append(firm["GreenCapital"][t] / total_capital if total_capital > 0 else 0)

            """            # --- Debugging
            if sim == 0 and t in [1, 5, 10]:
                print(f"[t={t}] Firm {firm['FirmID']}")
                print(f"  BrownInvDes={brown_inv_des:.3f}, GreenInvDes={green_inv_des:.3f}")
                print(f"  BrownLoansDem={firm['BrownLoansDem'][t]:.3f}, GreenLoansDem={firm['GreenLoansDem'][t]:.3f}")
                print(f"  BrownLoansVar={brown_loans_granted:.3f}, GreenLoansVar={green_loans_granted:.3f}")
                print(f"  BrownInvestment={brown_investment:.3f}, GreenInvestment={green_investment:.3f}")
                print(f"  CreditConstraintVar={cc_final:.3f}, allowed_brown_cap={allowed_brown:.3f}")
                print("-" * 60)
            """
           # À LA FIN DE CHAQUE FIRM, on peut calculer :
            bank = bank_list[firm["IdBank"]]
#            if t < len(firm["BrownCapital"]):
            all_results.append({
                "Simulation": sim,
                "Period": t,
                "FirmID": firm["FirmID"],
                "BrownCapital": firm["BrownCapital"][t],
                "GreenCapital": firm["GreenCapital"][t],
                "BrownInvestment": firm["BrownInvestment"][t],
                "GreenInvestment": firm["GreenInvestment"][t],
                "CreditConstraintVar": firm["CreditConstraintVar"][t],
                "TotalCapital": firm["TotalCapital"][t],
                "GreenCapitalRatio": firm["GreenCapitalRatio"][t],
                "Failures": bank["Failures"][0],
                "RiskAppetite": bank["RiskAppetite"][t],
                "TargetLeverage": bank["TargetLeverage"][t],
                "BanksAnimalSpirits": bank["BanksAnimalSpirits"][t],
                "ScenarioName": scenario_name,
                "Revenue": firm["Revenue"][t] if "Revenue" in firm and len(firm["Revenue"]) > t else 0.0,
                "AtmosphericTemperature": nature["AtmosphericTemperature"][t] if t < len(nature["AtmosphericTemperature"]) else np.nan,
            })


        # Moyenne du revenu disponible pour les ménages à t-1
        if t > 0:
            total_disp = sum(h["DisposableIncome"][t - 1] for h in household_list)
            mean_disp_income = total_disp / len(household_list)
        else:
            mean_disp_income = 1.0  # éviter division par zéro


        for household in household_list:
            if t == 0:
                household["DisposableIncome"].append(0)
                household["FinancialIncome"].append(0)
                household["TaxesInd"].append(0)
                household["HouseholdDebtService"].append(0)
                household["HouseholdNewDebt"].append(0)
                household["BaseConsumption"].append(0)
                household["Consumption"].append(0)
                household["Savings"].append(household["Savings"][0])

            household["w_needs"]     = random.uniform(0.1, 0.9)
            household["w_bandwagon"] = random.uniform(0.1, 0.9)
            household["w_inertia"]   = random.uniform(0.5, 1.0)  # on force une bonne inertie
            household["w_holistic"]  = random.uniform(0.1, 0.9)

            status = household["Status"]
            employer_id = household["IdEmployer"]
            savings_lag = household["Savings"][t - 1]
            debt_lag = household["HouseholdTotalDebt"][t - 1]
            income_lag = household["Income"][t - 1]

            # Revenu du ménage à la période t
            if status in [0, 4]:  # chômage "classique"
                last_wage = safe_get(household.get("Income", []), t - 1, 0.0)
                # Salaire de référence : dernier salaire si >0, sinon salaire minimum
                ref_wage = last_wage if last_wage > 0 else safe_get(minimum_wage, t, minimum_wage[0])
                income = UB_REPLACEMENT * ref_wage

            elif status == 3:  # emploi garanti (job guarantee)
                income = safe_get(minimum_wage, t, minimum_wage[0])

            else:
                firm = next((f for f in firm_list if f["FirmID"] == employer_id), None)
                if firm is not None:
                    if status == 1:
                        income = firm["LowSkilledBrownWage"][t]
                    elif status == 2:
                        income = firm["LowSkilledGreenWage"][t]
                    elif status == 5:
                        income = firm["HighSkilledBrownWage"][t]
                    elif status == 6:
                        income = firm["HighSkilledGreenWage"][t]
                    else:
                        income = 0.0
                else:
                    income = 0.0

            household["Income"].append(income)

            deposit_rate = household["DepositInterestRate"][0]
            financial_income = deposit_rate * savings_lag
            household["FinancialIncome"].append(financial_income)

            taxes = household["TaxRate"][0] * income_lag
            household["TaxesInd"].append(taxes)

            loan_rate = household["LoanRateCons"][0]
            debt_service = loan_rate * debt_lag
            household["HouseholdDebtService"].append(debt_service)

            red = household["Red"][0]
            disp_income = (income + financial_income + red) - (taxes + debt_service)
            household["DisposableIncome"].append(disp_income)

            # --- IMPACT DE LA TAXE CARBONE SUR LE REVENU DISPONIBLE DES MÉNAGES ---
            if carbon_tax_active:
                carbon_tax_rate = 0.1  # Exemple : 5% de taxe
                carbon_cost_household = carbon_tax_rate * household["DisposableIncome"][t]
                household["DisposableIncome"][t] = max(0, household["DisposableIncome"][t] - carbon_cost_household)

            # Paramètres de la politique sociale post growth
            RBU_amount = 0.15 * np.prod([(1 + general_inflation[i]) for i in range(t + 1)])
            # RBU_amount = 1  # Revenu de base universel fixe par ménage et période
            income_tax_threshold = 1  # Seuil de revenu à partir duquel la taxe s'applique
            progressive_tax_rate = 0.15  # Taux d'imposition sur la part de revenu > threshold

            if scenario_name == "post_growth":
                income_t = household["Income"][t]

                # Calcul de la taxe progressive simulée sur hauts revenus
                taxable_income = max(0, income_t - income_tax_threshold)
                tax_amount = progressive_tax_rate * taxable_income

                # Revenu disponible avant redistribution
                dispo_before = household["DisposableIncome"][t]

                # On retire la taxe
                dispo_after_tax = max(0, dispo_before - tax_amount)

                # Ajout du revenu de base universel
                dispo_final = dispo_after_tax + RBU_amount

                household["DisposableIncome"][t] = dispo_final

            # Calcul du coefficient de Gini sur le revenu disponible des ménages à la période t
            disposable_incomes_t = [h["DisposableIncome"][t] for h in household_list]
            gini_t = gini_coefficient(disposable_incomes_t)

            # Stockage de l’indicateur pour analyses ultérieures
            gini_records.append({
                "Simulation": sim,
                "Period": t,
                "GiniDisposableIncome": gini_t,
                "ScenarioName": scenario_name
            })

        # -- Prix sectoriels (inflation + pass-through de la taxe carbone) : MAJ 1x par t
        sector_names = ["Agriculture", "Energy", "Housing", "Transport", "Industry", "Technology"]
        carbon_tax_rate      = 0.05   # doit rester cohérent avec les autres blocs
        carbon_pass_through  = 1    # 100 % du coût de taxe répercuté dans les prix

        # 1) Capital brun agrégé par secteur à la période t
        brown_by_sector = {name: 0.0 for name in sector_names}
        total_brown = 0.0

        if CarbonTaxActive and len(CarbonTaxActive) > t and CarbonTaxActive[t] == 1:
            for f in firm_list:
                sid = int(f.get("IdSector", 0))
                if 1 <= sid <= 6:
                    brown_list = f.get("BrownCapital", [])
                    if not brown_list:
                        continue
                    if t < len(brown_list):
                        b_t = brown_list[t]
                    else:
                        b_t = brown_list[-1]
                    b_t = float(max(0.0, b_t))
                    sname = sector_names[sid - 1]
                    brown_by_sector[sname] += b_t
                    total_brown += b_t

        # 2) Surcroît d'inflation sectoriel dû à la taxe carbone
        extra_infl_by_sector = {name: 0.0 for name in sector_names}
        if CarbonTaxActive and len(CarbonTaxActive) > t and CarbonTaxActive[t] == 1 and total_brown > 0.0:
            for name in sector_names:
                share = brown_by_sector[name] / total_brown
                # plus un secteur est intensif en capital brun, plus le "cost-push" est élevé
                extra_infl_by_sector[name] = carbon_pass_through * carbon_tax_rate * share

        base_infl = general_inflation[t]

        # --- Revalorisation du salaire de base si scénario transition_mix ---
        if scenario_name == "transition_mix":
            # base_wage[t] = base_wage[t-1] * (1 + inflation_t)
            new_base_wage = base_wage_list[-1] * (1 + general_inflation[t])
        else:
            # Sinon, salaire de base constant
            new_base_wage = base_wage_list[-1]

        base_wage_list.append(new_base_wage)


        if t > 0:
            price_ag    = price_ag_list[t - 1]   * (1 + base_infl + extra_infl_by_sector["Agriculture"])
            price_ener  = price_ener_list[t - 1] * (1 + base_infl + extra_infl_by_sector["Energy"])
            price_hous  = price_hous_list[t - 1] * (1 + base_infl + extra_infl_by_sector["Housing"])
            price_trans = price_trans_list[t - 1]* (1 + base_infl + extra_infl_by_sector["Transport"])
            price_ind   = price_ind_list[t - 1]  * (1 + base_infl + extra_infl_by_sector["Industry"])
            price_tech  = price_tech_list[t - 1] * (1 + base_infl + extra_infl_by_sector["Technology"])
        else:
            # t = 0 : on garde les prix normalisés à 1.0 (pas de choc immédiat)
            price_ag    = 1.0
            price_ener  = 1.0
            price_hous  = 1.0
            price_trans = 1.0
            price_ind   = 1.0
            price_tech  = 1.0

        price_ag_list.append(price_ag)
        price_ener_list.append(price_ener)
        price_hous_list.append(price_hous)
        price_trans_list.append(price_trans)
        price_ind_list.append(price_ind)
        price_tech_list.append(price_tech)


        # -- Boucle ménages (UNE seule écriture de Consumption & SuppCons) — déplacée ici, non imbriquée
        for household in household_list:
            # === Initialisation des listes manquantes ===
            for key in [
                "ConsAg","ConsEner","ConsHous","ConsTrans",
                "BaseConsumption","Consumption","SuppCons",
                "SuppConsAg","SuppConsEner","SuppConsHous",
                "SuppConsTrans","SuppConsInd","SuppConsTC",
                "Savings","HouseholdNewDebt","HouseholdTotalDebt",
                "NeedsIndex","GrNeedsIndex"
            ]:
                if key not in household:
                    household[key] = []

            # === Étape t=0 ===
            if t == 0:
                household["ConsAg"].append(0.05)
                household["ConsEner"].append(0.01)
                household["ConsHous"].append(0.07)
                household["ConsTrans"].append(0.02)
                household["SuppCons"].append(0)
                for key in ["SuppConsAg","SuppConsEner","SuppConsHous","SuppConsTrans","SuppConsInd","SuppConsTC"]:
                    household[key].append(0.0)
                household["Savings"].append(0.0)
                household["HouseholdNewDebt"].append(0.0)
                household["HouseholdTotalDebt"].append(0.0)
                household["NeedsIndex"].append(100.0)
                household["GrNeedsIndex"].append(0.0)
                continue

            # === Étape t > 0 ===
            # 1) Base par catégorie (indexée sur variation de prix)
            cons_ag    = household["ConsAg"][t - 1]    * (1 + (price_ag    - price_ag_list[t - 1]))
            cons_ener  = household["ConsEner"][t - 1]  * (1 + (price_ener  - price_ener_list[t - 1]))
            cons_hous  = household["ConsHous"][t - 1]  * (1 + (price_hous  - price_hous_list[t - 1]))
            cons_trans = household["ConsTrans"][t - 1] * (1 + (price_trans - price_trans_list[t - 1]))

            household["ConsAg"].append(cons_ag)
            household["ConsEner"].append(cons_ener)
            household["ConsHous"].append(cons_hous)
            household["ConsTrans"].append(cons_trans)

            base_c = cons_ag + cons_ener + cons_hous + cons_trans
            household["BaseConsumption"].append(base_c)

            # 2) Décision de consommation totale
            disp_income = household["DisposableIncome"][t]
            savings_lag = household["Savings"][t - 1] if t > 0 else 0.0
            debt_lag    = household["HouseholdTotalDebt"][t - 1] if t > 0 else 0.0
            ptc_di      = float(household["PropensityToConsumeDI"][0])
            ptc_s       = float(household["PropensityToConsumeSavings"][0])

            available = disp_income + savings_lag

            supp_cons_desired   = max(0.0, ptc_di * (disp_income - base_c) + ptc_s * savings_lag)
            desired_consumption = base_c + supp_cons_desired

            if available >= desired_consumption:
                consumption = desired_consumption
                supp_cons   = supp_cons_desired
                household["HouseholdNewDebt"].append(0.0)
            elif available >= base_c:
                consumption = available
                supp_cons   = max(0.0, available - base_c)
                household["HouseholdNewDebt"].append(0.0)
            else:
                consumption = base_c
                supp_cons   = 0.0
                debt_needed = base_c - available
                household["HouseholdNewDebt"].append(debt_needed)

            household["Consumption"].append(consumption)
            household["SuppCons"].append(supp_cons)

            new_debt   = household["HouseholdNewDebt"][t]
            total_debt = debt_lag + new_debt
            household["HouseholdTotalDebt"].append(total_debt)

            saving = savings_lag + (disp_income - consumption + new_debt)
            household["Savings"].append(max(0.0, saving))

            # 3) Répartition sectorielle de la surconsommation
            household["SuppConsAg"].append(household["AgPref"]     * supp_cons)
            household["SuppConsEner"].append(household["EnerPref"] * supp_cons)
            household["SuppConsHous"].append(household["HousPref"] * supp_cons)
            household["SuppConsTrans"].append(household["TransPref"]* supp_cons)
            household["SuppConsInd"].append(household["IndPref"]   * supp_cons)
            household["SuppConsTC"].append(household["TCPref"]     * supp_cons)

            # 4) NeedsIndex
            hum0, hum1 = 0.01, 0.01
            disp_income_lag = household["DisposableIncome"][t - 1] if t > 0 else 1.0
            base_c_lag      = household["BaseConsumption"][t - 1] if t > 0 else 1.0
            needs_index_lag = household["NeedsIndex"][t - 1] if t > 0 else 100.0

            term1 = hum0 * (1 - base_c_lag / disp_income_lag) if disp_income_lag > 0 else 0.0
            term2 = hum1 * (disp_income_lag / mean_disp_income) if mean_disp_income > 0 else 0.0

            gr_needs_index = term1 + term2
            household["GrNeedsIndex"].append(gr_needs_index)
            new_needs_index = max(0.0, needs_index_lag * (1 + gr_needs_index))
            household["NeedsIndex"].append(new_needs_index)

            total_base = sum(
                h["BaseConsumption"][t]
                if "BaseConsumption" in h and len(h["BaseConsumption"]) > t
                else 0.0
                for h in household_list
            )

            total_supp = sum(
                h["SuppCons"][t]
                if "SuppCons" in h and len(h["SuppCons"]) > t
                else 0.0
                for h in household_list
            )

            total_conso = total_base + total_supp

            total_revenue= sum(f["Revenue"][t] if "Revenue" in f and len(f["Revenue"]) > t else 0.0 for f in firm_list)


#            print(f"[t={t}] Conso totale: {total_conso:.2f} | Revenus firmes: {total_revenue:.2f}")


            # Mise à jour fictive du vote pour permettre le calcul de l’effet bandwagon
            # (À remplacer plus tard par une vraie règle comportementale)
            prev_vote = household["VoteDecision"][-1]

            
            # --- DÉCISION DE VOTE ---

            needs_index = household["NeedsIndex"][t]
            needs_index_prev = household["NeedsIndex"][t - 1]
            vote_prev = household["VoteDecision"][t - 1]
            bandwagon_prev = bandwagon_effect[t - 1]

            # Pondérations
            w_needs     = household["w_needs"]
            w_bandwagon = household["w_bandwagon"]
            w_inertia   = household["w_inertia"]
            w_holistic  = household["w_holistic"]

            # --- Effet holiste x = 1 - Gini - (BanksProfits/GDP) ---

            # Gini courant (calculé plus haut dans la boucle sur t)
            gini_for_vote = gini_t if "gini_t" in locals() else 0.0

            # Profits bancaires sur PIB : on prend la période t-1 (les gens observent le passé)
            if t <= 0:
                banks_ratio = 0.0
            else:
                banks_ratio = get_banks_profits_to_gdp_ratio(t - 1)

            x = random.uniform(0.5, 1) - gini_for_vote - banks_ratio

            # --- Calcul de la probabilité de voter pro-transition ---

            # On combine besoins, bandwagon, inertie et effet holiste
            vote_raw = (
                w_needs     * (needs_index - needs_index_prev) +
                w_bandwagon * bandwagon_prev +
                w_inertia   * vote_prev +
                w_holistic  * x
            )

            # Passage par une logistique pour obtenir une probabilité entre 0 et 1
            vote_prob = 1.0 / (1.0 + math.exp(-vote_raw))

            # Décision finale
            new_vote = 1 if random.random() < vote_prob else 0
            household["VoteDecision"].append(new_vote)

#        if t in [5, 10, 15, 20, 24]:
 #           print(f"SIM {sim} | T = {t} | SCÉNARIO = {scenario_name}")
  #          print(f"→ Vote pro-transition : {sum(1 for h in household_list if h['VoteDecision'][t] == 1)}")
   #         print(f"→ TransitionActive[t] = {TransitionActive[t]}")
    #        print(f"→ CarbonTaxActive[t] = {CarbonTaxActive[t]}")
     #       print(f"→ PostGrowthActive[t] = {PostGrowthActive[t]}")



        for h_id, h in enumerate(household_list):
            needsindex_records.append({
                "Simulation": sim,
                "HouseholdID": h_id,
                "Period": t,
                "NeedsIndex": h["NeedsIndex"][t],
                "ScenarioName": scenario_name
            })


        involuntary_unemployed = sum(1 for h in household_list if h["Status"] == 3 and h["IdEmployer"] == -1)
        unemployment_rate = involuntary_unemployed / num_households

        social_cost_records.append({
        "Simulation": sim,
        "Period": t,
        "UnemploymentRate": unemployment_rate,
        "ScenarioName": scenario_name
        })

        status_counts = {s: 0 for s in range(8)}
        for h in household_list:
                    # Sécurisation explicite
            for key in ["SuppCons", "BaseConsumption", "Consumption"]:
                if len(household[key]) <= t:
                    household[key].append(0.0)
            for key in ["SuppConsAg", "SuppConsEner", "SuppConsHous", "SuppConsTrans", "SuppConsInd", "SuppConsTC"]:
                if len(household[key]) <= t:
                    household[key].append(0.0)


            s = h["Status"]
            status_counts[s] += 1

        for s in range(8):
            status_share_records.append({
                "Simulation": sim,
                "Period": t,
                "Status": s,
                "Share": status_counts[s] / num_households,
                "ScenarioName": scenario_name
            })

        if scenario_name == "carbon_tax_only":
            policy_active = CarbonTaxActive[t] if len(CarbonTaxActive) > t else 0
        elif scenario_name == "transition_mix":
            policy_active = TransitionActive[t] if len(TransitionActive) > t else 0
        elif scenario_name == "post_growth":
            policy_active = PostGrowthActive[t] if len(PostGrowthActive) > t else 0
        else:
            policy_active = 0  # fallback de sécurité

        policy_outcomes.append({
            "Simulation": sim,
            "ScenarioName": scenario_name,
            "Period": t,
            "EcologicalPolicyActive": policy_active
        })

    # --- CALCUL DU PIB ---

    # 1. Consommation totale
#        total_consumption = sum(h["Consumption"][t] for h in household_list)


    # 2. Investissement total (brown + green) des firmes
        total_investment = sum(
            (f["BrownInvestment"][t] if "BrownInvestment" in f and len(f["BrownInvestment"]) > t else 0)
            +
            (f["GreenInvestment"][t] if "GreenInvestment" in f and len(f["GreenInvestment"]) > t else 0)
            for f in firm_list
        )
                
        print(f"[Période {t}] Investissement agrégé : "
            f"Total={total_investment:.3f}")
        print("=" * 80)

        """         # 3. Dépenses publiques :
        # - investissement public vert (part de GreenInvestment non privée)
        public_green_investment = 0.0
        if t > 0 and TransitionActive[t - 1] == 1:
            for f in firm_list:
                green_capital_last = f["GreenCapital"][t - 1]
                public_green_investment += 0.05 * green_capital_last  # taux codé en dur

        # - allocations chômage (Status = 0, 3, 4)
        public_unemployment_benefits = sum(unemployment_benefit[t] for h in household_list if h["Status"] in [0, 4])

        # - emploi garanti (Status = 3 sans employeur)
        public_guaranteed_jobs = sum(minimum_wage[t] for h in household_list if h["Status"] == 3 and h["IdEmployer"] == -1)
        """


        # === Dépenses publiques (version sans capage, avec subventions vertes et UB liés aux salaires) ===

        # Paramètres globaux (à définir en haut du script)
        SUBSIDY_RATE_GREEN = 0.10    # ex: 20 % de co-financement public de l'investissement vert

        # 1) Subventions vertes (flux, pas stock)
        public_green_investment = 0.0
        if t > 0 and safe_get(TransitionActive, t-1, 0) == 1:
            for f in firm_list:
                gi_flux = safe_get(f.get("GreenInvestment", []), t-1, 0.0)
                public_green_investment += SUBSIDY_RATE_GREEN * gi_flux

        # 2) Allocations chômage (proportion du dernier salaire ou salaire minimum)
        public_unemployment_benefits = 0.0
        for h in household_list:
            status = h["Status"]
            if status in [0, 4]:  # chômeur
                last_wage = safe_get(h.get("Income", []), t-1, 0.0)
                ref_wage  = last_wage if last_wage > 0 else safe_get(minimum_wage, t, 0.0)
                public_unemployment_benefits += UB_REPLACEMENT * ref_wage

        # 3) Emplois garantis (tous les statuts 3 sans employeur reçoivent un salaire minimum complet)
        mw_t = safe_get(minimum_wage, t, 0.0)
        public_guaranteed_jobs = sum(mw_t for h in household_list if h["Status"] == 3 and h["IdEmployer"] == -1)

        # 4) Total G (SFC)
        total_public_spending = public_green_investment + public_unemployment_benefits + public_guaranteed_jobs


        for f in firm_list:
            f.setdefault("Sales", [])
            while len(f["Sales"]) <= t:
                f["Sales"].append(0)
            f["Sales"][t] = 0
            # idem pour les ventes par produit
            for p in f["Products"]:
                p.setdefault("Sales_Prod", [])
                while len(p["Sales_Prod"]) <= t:
                    p["Sales_Prod"].append(0)
                p["Sales_Prod"][t] = 0



        update_firm_prices_from_sector_lists(
            firm_list, t,
            price_ag_list, price_ener_list,
            price_hous_list, price_trans_list,
            price_ind_list, price_tech_list
        )
        
        update_firm_characteristics_from_products(firm_list, t)

        update_household_consumption(household_list, firm_list, t, params, num_sectors=6)

        update_firm_budget_vectorized(firm_list, t, paramsinnov)

        firm_list = update_innovation_and_adoption(firm_list, household_list, t, paramsinnov)

        # === Debug : compter les changements de supplier ===
        changes = 0
        for hh in household_list:
            if len(hh["FirmID"]) > 1:  # au moins deux périodes enregistrées
                if hh["FirmID"][-1] != hh["FirmID"][-2]:  # si le supplier a changé
                    changes += 1

        print(f"Période {t} : {changes} ménages ont changé de supplier.")


        print(
            f"[Période {t}] Innovation active : "
            f"{sum(1 for f in firm_list if f.get('Portfolio', [0])[-1] > 0)} firmes ont adopté le produit 2"
        )

        # =========================================================
        # === PIPELINE CONSISTENT POUR UNE SEULE PÉRIODE t ========
        # =========================================================

        # 1) Conso sectorielle de base (ancienne logique)
        sector_revenue_base = compute_sector_revenue_base(t, household_list)

        # Flags scénarios à t (robustes)
        is_tm = 1 if (len(TransitionActive) > t and TransitionActive[t] == 1) else 0
        is_pg = 1 if (len(PostGrowthActive)  > t and PostGrowthActive[t]  == 1) else 0

        # 2) Suffisance : ACTIVE seulement si Transition OU PostGrowth actifs
        if is_tm or is_pg:
            sector_revenue_market = apply_sufficiency_module(
                t, sector_revenue_base, household_list,
                purchase_freq=globals().get("purchase_freq"),
                used_params=globals().get("used_params"),
                buyers_share_at_t=globals().get("buyers_share_at_t"),   # OK ici, car module actif
                used_share_for_sector=globals().get("used_share_for_sector"),
                antiobs_multiplier_new_value=globals().get("antiobs_multiplier_new_value"),
                New_value_by_sector=globals().get("New_value_by_sector"),
                Repair_value_by_sector=globals().get("Repair_value_by_sector"),
            )
        else:
            # carbon_tax_only → PAS de suffisance
            sector_revenue_market = dict(sector_revenue_base)

        # === 3) Socialisation simple (PostGrowth et, optionnellement, Transition) ===
        sector_after_social, G_socialized_t = apply_socialization_module(
            t,
            sector_revenue_market,
            PostGrowthActive=PostGrowthActive,
            TransitionActive=TransitionActive,
            pg_cons=pg_cons,
            soft_factor_tm=0,  # ou 0.0 si tu veux aucune socialisation en Transition
        )

        # === 4) Calcul des agrégats macroéconomiques ===
        CTot_t = float(sum(sector_after_social.values()))  # conso de marché après socialisation
        GTot_t = float(
            public_green_investment
            + public_unemployment_benefits
            + public_guaranteed_jobs
            + G_socialized_t
        )
        ITot_t = total_investment

        # --- Suivi explicite du G partagé ---
        G_shared_list[t] = G_socialized_t

        # === (1) Récupération des agrégats au t (adapte les noms si besoin) ===
        g_shared_t = G_shared_list[t] if 'G_shared_list' in globals() and len(G_shared_list) > t else 0.0
        i_pub_shared_t = I_pub_shared_list[t] if 'I_pub_shared_list' in globals() and len(I_pub_shared_list) > t else 0.0
        total_public_spending_t = total_public_spending + g_shared_t + i_pub_shared_t

        # Revenu agrégé des ménages (adapte "Income" si ta clé est différente)
        household_income_agg_t = 0.0
        for h in household_list:
            if "Income" in h and len(h["Income"]) > t:
                household_income_agg_t += h["Income"][t] or 0.0

        # Profits agrégés des firmes
        firms_profits_agg_t = 0.0
        for f in firm_list:
            if "Profits" in f and len(f["Profits"]) > t:
                firms_profits_agg_t += f["Profits"][t] or 0.0

        # Recettes de taxe carbone agrégées (si tu ne les stockes pas déjà au niveau État)
        carbon_tax_revenue_t = compute_carbon_tax_revenue_t(t, firm_list)

        # Inflation & PIB
        inflation_t = general_inflation[t] if len(general_inflation) > t else 0.0
        gdp_t = (
            GDP[t] if 'GDP' in globals() and len(GDP) > t
            else (gdp_records[t]['GDP'] if 'gdp_records' in globals() and len(gdp_records) > t and 'GDP' in gdp_records[t] else 0.0)
        )

        # === (2) Mise à jour Gouvernement & Banque centrale ===
        update_government_and_cb_accounts(
            t,
            total_public_spending_t=total_public_spending_t,
            g_shared_t=g_shared_t,
            i_pub_shared_t=i_pub_shared_t,
            household_income_agg_t=household_income_agg_t,
            firms_profits_agg_t=firms_profits_agg_t,
            carbon_tax_revenue_t=carbon_tax_revenue_t,  # <-- taxe carbone = RECETTE de l’État
            inflation_t=inflation_t,
            gdp_t=gdp_t,
            gdp_trend_t=gdp_t  # provisoire : on prend le PIB courant comme tendance
        )


        # === 5) Enregistrement des agrégats dans le GDP record ===
        gdp_records.append({
            "Period": t,
            "ScenarioName": scenario_name,
            "CTot": CTot_t,
            "GTot": GTot_t,
            "ITot": ITot_t,
            "GDP": CTot_t + GTot_t + ITot_t,
            "G_shared": G_socialized_t,
        })

        # === Enregistrement dette publique (mêmes clés que gdp_records : Period, Scenario) ===
        # On prend le PIB courant tel qu'il vient d'être calculé (cohérent avec gdp_records)
        gdp_current_t = CTot_t + GTot_t + ITot_t

        # Valeurs gouvernement / BC / banques à t (robustes si la liste est courte)
        debt_t      = Government["PublicDebt"][t] if len(Government["PublicDebt"]) > t else 0.0
        igov_raw    = Government["i_gov"][t]      if len(Government["i_gov"])      > t else None
        cb_rate_t   = CentralBank["cb_rate"][t]   if len(CentralBank["cb_rate"])   > t else 0.0

        # Si i_gov n'a pas été apposé dans Government, on reconstruit comme dans update_government_and_cb_accounts
        if igov_raw is None:
            debt_ratio_t = (debt_t / gdp_current_t) if gdp_current_t > 0 else 0.0
            igov_t = max(cb_rate_t + 0.005 * (1.0 + debt_ratio_t), 0.0)
        else:
            igov_t = igov_raw

        public_debt_records.append({
            "Period": t,
            "ScenarioName": scenario_name,          # idem gdp_records
            "PublicDebt": debt_t,               # montant total détenu
            "DebtToGDP_pct": 100.0 * (debt_t / gdp_current_t) if gdp_current_t > 0 else 0.0,
            "i_gov_pct": 100.0 * igov_t
        })


        # === Construction finale de gdp_df ===
        gdp_df = pd.DataFrame(gdp_records)

        # === Construction du DataFrame dette publique (même style que gdp_df) ===
        public_debt_df = pd.DataFrame(public_debt_records)

        # (Optionnel) filtrer t=0 dans les graphiques, comme pour le PIB
        public_debt_plot = public_debt_df[public_debt_df["Period"] > 0]

        cols = ["GDP","CTot","ITot","GTot","G_shared"]
        agg = (gdp_df.groupby(["ScenarioName","Period"], as_index=False)[cols]
                    .mean()
                    .sort_values(["ScenarioName","Period"]))
        agg["GDP_growth_pct"] = (agg.groupby("ScenarioName")["GDP"].pct_change()
                                .replace([np.inf,-np.inf], np.nan)
                                .fillna(0.0) * 100)

        policy_df = pd.DataFrame(policy_outcomes)

        eco_share = (
            policy_df.groupby(["ScenarioName", "Period"])["EcologicalPolicyActive"]
                    .mean()
                    .reset_index()
                    .rename(columns={"EcologicalPolicyActive": "EcoPolicyActiveShare"})
        )

        agg = agg.merge(eco_share, on=["ScenarioName", "Period"], how="left")
        agg["EcoPolicyActiveShare"] = agg["EcoPolicyActiveShare"].fillna(0.0)


        print(agg[["ScenarioName","Period","EcoPolicyActiveShare",
                "GDP","GDP_growth_pct","CTot","ITot","GTot","G_shared"]]
            .to_string(index=False, justify="center"))


        update_biophys_vars(nature, firm_list, t, params)
        

        # --- MISE À JOUR DES POLITIQUES PAR ÉLECTIONS (sans append) ---
        if t == 0:
            TransitionActive[t] = 0
            CarbonTaxActive[t]  = 0
            PostGrowthActive[t] = 0

        elif t in ELECTIONS:
            # Prolonge l'état courant pour t (pas de saut à t)
            TransitionActive[t] = TransitionActive[t-1]
            CarbonTaxActive[t]  = CarbonTaxActive[t-1]
            PostGrowthActive[t] = PostGrowthActive[t-1]

            # Les votes au pas t DOIVENT déjà exister
            vote = majority_wants_policy(t)   # -> True / False / None (si votes incomplets)

            start = t + 1
            stop  = min(t + MANDATE_LEN, num_periods - 1)   # inclusif
            if start <= stop:
                L = stop - start + 1

                if vote is None:
                    # Votes incomplets → on prolonge l'état courant
                    TransitionActive[start:stop+1] = [TransitionActive[t-1]] * L
                    CarbonTaxActive[start:stop+1]  = [CarbonTaxActive[t-1]]  * L
                    PostGrowthActive[start:stop+1] = [PostGrowthActive[t-1]] * L

                elif vote is True:
                    if scenario_name == "carbon_tax_only":
                        TransitionActive[start:stop+1] = [0] * L
                        CarbonTaxActive[start:stop+1]  = [1] * L
                        PostGrowthActive[start:stop+1] = [0] * L
                    elif scenario_name == "transition_mix":
                        TransitionActive[start:stop+1] = [1] * L
                        CarbonTaxActive[start:stop+1]  = [1] * L
                        PostGrowthActive[start:stop+1] = [0] * L
                    elif scenario_name == "post_growth":
                        TransitionActive[start:stop+1] = [0] * L   
                        CarbonTaxActive[start:stop+1]  = [0] * L
                        PostGrowthActive[start:stop+1] = [1] * L
                        # Semis de K_shared la première fois qu'on active PostGrowth
                        if scenario_name == "post_growth" and not seeded_shared:
                            seed_initial_shared_capital_from_private0(
                                firm_list, K_shared, seed_ratio_by_sector, scenario_name, PostGrowthActive
                            )
                            seeded_shared = True
                    else:
                        # Scénario inconnu → prolongation
                        TransitionActive[start:stop+1] = [TransitionActive[t-1]] * L
                        CarbonTaxActive[start:stop+1]  = [CarbonTaxActive[t-1]]  * L
                        PostGrowthActive[start:stop+1] = [PostGrowthActive[t-1]] * L

                else:
                    # Rejet → tout à 0 sur la fenêtre
                    TransitionActive[start:stop+1] = [0] * L
                    CarbonTaxActive[start:stop+1]  = [0] * L
                    PostGrowthActive[start:stop+1] = [0] * L

        else:
            # Pas d'élection → prolonger l'état courant
            TransitionActive[t] = TransitionActive[t-1]
            CarbonTaxActive[t]  = CarbonTaxActive[t-1]
            PostGrowthActive[t] = PostGrowthActive[t-1]

        # Indicateur synthétique AU PAS t
        if scenario_name == "carbon_tax_only":
            ecological_policy_active = int(CarbonTaxActive[t] == 1)
        elif scenario_name == "transition_mix":
            ecological_policy_active = int(TransitionActive[t] == 1)
        elif scenario_name == "post_growth":
            ecological_policy_active = int(PostGrowthActive[t] == 1)
        else:
            ecological_policy_active = 0

        current_scenario = scenario_name

        # Log APRÈS mise à jour (et avant l'économie réelle)
        policy_records.append({
            "Simulation": sim,
            "ScenarioName": scenario_name,
            "Period": t,
            "EcologicalPolicyActive": ecological_policy_active,
            "CarbonTaxActive": CarbonTaxActive[t],
            "TransitionActive": TransitionActive[t],
            "PostGrowthActive": PostGrowthActive[t]
        })



        # Création d'un DataFrame pour VoteDecision
        for h_id, h in enumerate(household_list):
            for t in range(1, num_periods):
                if t < len(h["VoteDecision"]):
                    vote_records.append({
                        "HouseholdID": h_id,
                        "Period": t,
                        "VoteDecision": h["VoteDecision"][t],
                        "ScenarioName": scenario_name,  # ajout
                        "Simulation": sim
                    })
        

for h_id, h in enumerate(household_list):
    for t in range(1, num_periods):
        household_records.append({
            "HouseholdID": h_id,
            "IdEmployer": h["IdEmployer"],
            "Period": t,
            "Income": h["Income"][t],
            "DisposableIncome": h["DisposableIncome"][t],
            "BaseConsumption": h["BaseConsumption"][t],
            "Consumption": h["Consumption"][t],
            "Savings": h["Savings"][t],
            "Debt": h["HouseholdTotalDebt"][t],
            "VoteDecision": h["VoteDecision"][t],
            "ScenarioName": current_scenario,  #  Ajout
            "Simulation": sim                  #  Ajout
        })

gini_df = pd.DataFrame(gini_records)

plt.figure(figsize=(10, 6))
sns.lineplot(data=gini_df, x="Period", y="GiniDisposableIncome", hue="ScenarioName", estimator="mean", ci="sd")
plt.title("Évolution moyenne du coefficient de Gini du revenu disponible par scénario")
plt.xlabel("Période")
plt.ylabel("Coefficient de Gini")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Visualisation du PIB par scénario (avec écart-type) ---
plt.figure(figsize=(10, 6))
filtered = gdp_df[gdp_df["ScenarioName"].isin(["carbon_tax_only", "transition_mix", "post_growth"])]
sns.lineplot(
    data=filtered, x="Period", y="GDP",
    hue="ScenarioName", estimator="mean", ci="sd"
)
plt.title("PIB moyen par scénario (avec écart-type)")
plt.xlabel("Période")
plt.ylabel("PIB agrégé")
plt.grid(True)
plt.tight_layout()
plt.show()

# 1) Aperçu comparé par période/scénario (moyenne sur runs)
print("\n=== PublicDebt (moyenne) par scénario/période ===")
print(public_debt_df.groupby(["ScenarioName","Period"])["PublicDebt"].mean().unstack(0).head(12).round(2))

# 2) Même chose pour DebtToGDP_pct pour comparer
print("\n=== DebtToGDP_pct (moyenne) par scénario/période ===")
print(public_debt_df.groupby(["ScenarioName","Period"])["DebtToGDP_pct"].mean().unstack(0).head(12).round(2))


# (1) Montant de dette publique
plt.figure(figsize=(10,6))
sns.lineplot(
    data=public_debt_plot,
    x="Period", y="PublicDebt", hue="ScenarioName",
    estimator="mean", ci="sd"
)
plt.title("Dette publique – moyenne par scénario (±1σ)")
plt.xlabel("Période"); plt.ylabel("Montant")
plt.grid(True); plt.tight_layout(); plt.show()

# (2) Ratio dette/PIB (%)
plt.figure(figsize=(10,6))
sns.lineplot(
    data=public_debt_plot,
    x="Period", y="DebtToGDP_pct", hue="ScenarioName",
    estimator="mean", ci="sd"
)
plt.title("Dette publique / PIB (%) – moyenne par scénario (±1σ)")
plt.xlabel("Période"); plt.ylabel("% du PIB")
plt.grid(True); plt.tight_layout(); plt.show()

# (3) Taux moyen payé sur la dette publique (%)
plt.figure(figsize=(10,6))
sns.lineplot(
    data=public_debt_plot,
    x="Period", y="i_gov_pct", hue="ScenarioName",
    estimator="mean", ci="sd"
)
plt.title("Taux d’intérêt moyen sur dette publique (i_gov, %) – moyenne par scénario (±1σ)")
plt.xlabel("Période"); plt.ylabel("%")
plt.grid(True); plt.tight_layout(); plt.show()


# --- À la fin de la simulation ---
num_firms_alive = [sum(1 for f in firm_list if f["Dead"][t] == 0) for t in range(num_periods)]
num_firms_p2 = [
    sum(1 for f in firm_list if len(f["Portfolio"]) > 0 and f["Portfolio"][-1] >= 1)
    for t in range(num_periods)
]

avg_X_P2 = [np.mean([f["X_P2"][t] for f in firm_list if len(f["X_P2"]) > t]) for t in range(num_periods)]
avg_Eff_P2 = [np.mean([f["Eff_P2"][t] for f in firm_list if len(f["Eff_P2"]) > t]) for t in range(num_periods)]
avg_Tox_P2 = [np.mean([f["Tox_P2"][t] for f in firm_list if len(f["Tox_P2"]) > t]) for t in range(num_periods)]
avg_Bio_P2 = [np.mean([f["Bio_P2"][t] for f in firm_list if len(f["Bio_P2"]) > t]) for t in range(num_periods)]

        # --- Contrôle socialisation (fin de simulation) — version robuste ---
def _safe_last(lst, default=0.0):
    return lst[-1] if isinstance(lst, list) and len(lst) > 0 else default

cons_brut_series = []
cons_market_series = []

for tt in range(num_periods):
            # Somme des conso ménages en protégeant les index
    c_tt = 0.0
    for h in household_list:
        cons_list = h.get("Consumption", [])
        c_tt += safe_get(cons_list, tt, _safe_last(cons_list, 0.0))
    cons_brut_series.append(c_tt)

    G_shared_tt = safe_get(G_shared_list, tt, 0.0)
    cons_market_series.append(max(0.0, c_tt - G_shared_tt))


plt.figure(figsize=(8,5))
plt.plot(range(num_periods), cons_brut_series, label="C (ménages, brut)")
plt.plot(range(num_periods), G_shared_list[:num_periods] if isinstance(G_shared_list, list) else [0.0]*num_periods,
                label="G_shared (opex socialisé)")
plt.plot(range(num_periods), cons_market_series, label="C_market (après socialisation)")
plt.title("Bascule C → G (socialisation post-croissance)")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# --- Graphiques ---
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.plot(range(num_periods), num_firms_alive, label="Firmes actives")
plt.plot(range(num_periods), num_firms_p2, label="Firmes avec P2 adopté")
plt.xlabel("Période")
plt.ylabel("Nombre de firmes")
plt.legend()
plt.title("Adoption et survie des firmes")

plt.subplot(1,2,2)
plt.plot(range(num_periods), avg_X_P2, label="X_P2 moyen")
plt.plot(range(num_periods), avg_Eff_P2, label="Eff_P2 moyen")
plt.plot(range(num_periods), avg_Tox_P2, label="Tox_P2 moyen")
plt.plot(range(num_periods), avg_Bio_P2, label="Bio_P2 moyen")
plt.xlabel("Période")
plt.ylabel("Valeur moyenne")
plt.legend()
plt.title("Évolution des caractéristiques du produit 2")

plt.tight_layout()
plt.show()

household_df = pd.DataFrame(household_records)

results_df = pd.DataFrame(all_results)

social_cost_df = pd.DataFrame(social_cost_records)

status_df = pd.DataFrame(status_share_records)

# === Visualisation du NeedsIndex par scénario ===

needsindex_df = pd.DataFrame(needsindex_records)


vote_df = pd.DataFrame(vote_records)



plot_gdp_components(gdp_df)

plot_atmospheric_temperature(results_df)



max_period = household_df["Period"].max()
subset = household_df[household_df["Period"] <= max_period]


vote_share_by_sim = (
    vote_df.groupby(["ScenarioName","Simulation","Period"])["VoteDecision"]
           .mean()
           .reset_index()
)

vote_summary = (
    vote_share_by_sim.groupby(["ScenarioName","Period"])["VoteDecision"]
                     .agg(Mean="mean", Std="std")
                     .reset_index()
)
# puis tracer toujours avec Mean / Std


# 1) Récupérer ou construire policy_df
try:
    policy_df
except NameError:
    policy_df = pd.DataFrame(policy_records)

# Sanity check
needed = {"ScenarioName", "Simulation", "Period", "EcologicalPolicyActive"}
missing = needed - set(policy_df.columns)
if missing:
    raise RuntimeError(f"Colonnes manquantes dans policy_df: {missing}")

policy_df["EcologicalPolicyActive"] = policy_df["EcologicalPolicyActive"].astype(int)

# 2) Somme des périodes actives par (scénario, simulation)
active_by_sim = (
    policy_df.groupby(["ScenarioName", "Simulation"], as_index=False)["EcologicalPolicyActive"]
    .sum()
    .rename(columns={"EcologicalPolicyActive": "ActivePeriods"})
)

# 3) Récap par scénario
summary = (
    active_by_sim.groupby("ScenarioName", as_index=False)["ActivePeriods"]
    .agg(MeanActivePeriods="mean", StdActivePeriods="std", MinActivePeriods="min", MaxActivePeriods="max")
)

totals = (
    policy_df.groupby("ScenarioName", as_index=False)["EcologicalPolicyActive"]
    .sum()
    .rename(columns={"EcologicalPolicyActive": "TotalActivePeriods"})
)
summary = summary.merge(totals, on="ScenarioName", how="left")

# 4) Plot simple: moyenne des périodes actives (barres)
plt.figure(figsize=(8, 5))
x = np.arange(len(summary))
y = summary["MeanActivePeriods"].values
plt.bar(x, y)
plt.xticks(x, summary["ScenarioName"].tolist())
plt.ylabel("Périodes actives (moyenne par simulation)")
plt.title("Nombre de périodes où la politique est active (par scénario)")

# Annotations
ymax = (y.max() if len(y) else 0) or 1.0
for i, v in enumerate(y):
    plt.text(i, v + 0.03*ymax, f"{v:.1f}", ha="center", va="bottom")

plt.tight_layout()
plt.show()

# 5) (Option) un deuxième plot: total des périodes actives (somme sur toutes les simulations)
tot = summary[["ScenarioName", "TotalActivePeriods"]].sort_values("ScenarioName")
plt.figure(figsize=(8, 5))
x2 = np.arange(len(tot))
plt.bar(x2, tot["TotalActivePeriods"].values)
plt.xticks(x2, tot["ScenarioName"].tolist())
plt.ylabel("Total de périodes actives (toutes simulations)")
plt.title("Total de périodes avec politique active (par scénario)")
for i, v in enumerate(tot["TotalActivePeriods"].values):
    plt.text(i, v + 0.02*(tot["TotalActivePeriods"].max() or 1.0), f"{int(v)}", ha="center", va="bottom")
plt.tight_layout()
plt.show()

# 6) Petit tableau récap
print(
    summary.assign(
        MeanActivePeriods=lambda d: d["MeanActivePeriods"].round(2),
        StdActivePeriods=lambda d: d["StdActivePeriods"].round(2),
    ).sort_values("ScenarioName").to_string(index=False)
)

# Agrégation par période
revenue_agg = results_df.groupby("Period")["Revenue"].sum().reset_index(name="RevenusFirme")
consumption_agg = household_df.groupby("Period")["Consumption"].sum().reset_index(name="ConsommationMenages")

# Fusion des deux
merged_df = pd.merge(revenue_agg, consumption_agg, on="Period")

# Tracé
plt.figure(figsize=(10, 6))
plt.plot(merged_df["Period"], merged_df["ConsommationMenages"], label="Consommation totale des ménages", linewidth=2)
plt.plot(merged_df["Period"], merged_df["RevenusFirme"], label="Revenus totaux des firmes", linewidth=2, linestyle='--')
plt.xlabel("Période")
plt.ylabel("Montants")
plt.title("Consommation des ménages vs Revenus des firmes")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Tracé
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=subset,
    x="Period",
    y="IdEmployer",
    hue="HouseholdID",
    palette="husl",
    linewidth=0.8,
    legend=False,
    alpha=0.6
)

plt.title("Évolution de l'IdEmployer des ménages au cours du temps")
plt.xlabel("Période")
plt.ylabel("IdEmployer (-1 = sans emploi)")
plt.grid(True)
plt.tight_layout()
plt.show()

# On suppose que vote_df est déjà bien formaté
# Groupement par scénario, période et simulation pour calculer les parts
vote_agg = (
    vote_df.groupby(["ScenarioName", "Simulation", "Period"])["VoteDecision"]
    .mean()
    .reset_index()
)

# Agrégation finale : moyenne et écart-type par scénario et période
vote_summary = (
    vote_agg.groupby(["ScenarioName", "Period"])["VoteDecision"]
    .agg(Mean="mean", Std="std")
    .reset_index()
)

# Graphique
plt.figure(figsize=(12, 6))
for scenario in vote_summary["ScenarioName"].unique():
    data = vote_summary[vote_summary["ScenarioName"] == scenario]
    plt.plot(data["Period"], data["Mean"], label=scenario)
    plt.fill_between(
        data["Period"],
        data["Mean"] - data["Std"],
        data["Mean"] + data["Std"],
        alpha=0.2
    )

plt.title("Proportion de VoteDecision = 1 par période (avec écart-type)")
plt.xlabel("Période")
plt.ylabel("Proportion pro-transition")
plt.ylim(0, 1)
plt.legend(title="Scénario")
plt.grid(True)
plt.tight_layout()
plt.show()


# Si ce n'est pas déjà fait
# vote_df = pd.DataFrame(vote_records)

# Ajout de l'information de simulation si absente (utile si vote_df n'a pas "Simulation")
if "Simulation" not in vote_df.columns:
    household_simulation_map = household_df[["HouseholdID", "Simulation"]].drop_duplicates()
    vote_df = vote_df.merge(household_simulation_map, on="HouseholdID")

# Calcul des parts pro-transition par scénario, simulation, période
vote_share_by_sim = (
    vote_df.groupby(["ScenarioName", "Simulation", "Period"])["VoteDecision"]
    .mean()
    .reset_index()
)

# Moyenne et écart-type par scénario et période
vote_summary = (
    vote_share_by_sim
    .groupby(["ScenarioName", "Period"])["VoteDecision"]
    .agg(MeanVoteShare="mean", StdDevVoteShare="std")
    .reset_index()
)

results_df["GreenCapitalShare"] = results_df["GreenCapital"] / results_df["TotalCapital"]

# 3. Part du capital vert
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x="Period", y="GreenCapitalShare", hue="ScenarioName", estimator="mean", ci="sd")
plt.title("Part du capital vert dans le capital total")
plt.tight_layout()
plt.show()

# Tracé
plt.figure(figsize=(12, 6))
for scenario in vote_summary["ScenarioName"].unique():
    data = vote_summary[vote_summary["ScenarioName"] == scenario]
    plt.plot(data["Period"], data["MeanVoteShare"], label=scenario)
    plt.fill_between(
        data["Period"],
        data["MeanVoteShare"] - data["StdDevVoteShare"],
        data["MeanVoteShare"] + data["StdDevVoteShare"],
        alpha=0.2
    )

plt.title("Proportion moyenne de votes pro-transition par période et scénario")
plt.xlabel("Période")
plt.ylabel("Part moyenne de VoteDecision = 1")
plt.ylim(0, 1)
plt.legend(title="Scénario")
plt.grid(True)
plt.tight_layout()
plt.show()


# Évolution agrégée des votes pro-transition
plt.figure(figsize=(10, 5))
vote_share = vote_df.groupby("Period")["VoteDecision"].mean()
plt.plot(vote_share.index, vote_share.values, marker="o")
plt.title("Part des votes pro-transition (VoteDecision = 1) dans la population")
plt.xlabel("Période")
plt.ylabel("Part de votes pro-transition")
plt.grid(True)
plt.tight_layout()
plt.show()

policy_count_df = (
    policy_df.groupby(["ScenarioName", "Period"])["EcologicalPolicyActive"]
    .sum()
    .reset_index()
)

# 2. Ajouter le scénario associé à chaque ménage
household_scenario_map = household_df[["HouseholdID", "ScenarioName"]].drop_duplicates()
needsindex_df_full = needsindex_df.merge(household_scenario_map, on="HouseholdID")

policy_df = pd.DataFrame(policy_records)

policy_count_df = (
    policy_df.groupby(["ScenarioName", "Period"])["EcologicalPolicyActive"]
    .sum()
    .reset_index()
)

plt.figure(figsize=(10, 6))
sns.lineplot(
    data=policy_count_df,
    x="Period",
    y="EcologicalPolicyActive",
    hue="ScenarioName",
    marker="o"
)
plt.title("Nombre de simulations avec politique écologique active (par période et scénario)")
plt.xlabel("Période")
plt.ylabel("Nombre de simulations (sur 10)")
plt.ylim(0, 10)
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# Filtrage facultatif pour n'afficher qu'un seul scénario à la fois
for scenario in policy_df["ScenarioName"].unique():
    scenario_subset = policy_df[policy_df["ScenarioName"] == scenario]

    # Pivot table : lignes = simulations, colonnes = périodes, valeurs = active (0 ou 1)
    heatmap_data = scenario_subset.pivot_table(
        index="Simulation",
        columns="Period",
        values="EcologicalPolicyActive",
        fill_value=0
    )

    plt.figure(figsize=(12, 4))
    sns.heatmap(heatmap_data, cmap="Greens", cbar=True, linewidths=0.2, linecolor="gray", square=False)
    plt.title(f"Périodes avec politique écologique active – scénario : {scenario}")
    plt.xlabel("Période")
    plt.ylabel("Simulation")
    plt.tight_layout()
    plt.show()

# === Visualisation du NeedsIndex par scénario ===

# Assure-toi que needsindex_df est bien créé avec ScenarioName dans needsindex_records
# (ce bloc suppose que c’est déjà le cas)

# Agrégation par moyenne des parts sur toutes les simulations
status_df_grouped = status_df.groupby(["ScenarioName", "Period", "Status"], as_index=False)["Share"].mean()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=policy_count_df,
    x="Period",
    y="EcologicalPolicyActive",
    hue="ScenarioName"
)
plt.title("Nombre de simulations avec victoire politique écologique par scénario")
plt.xlabel("Période électorale")
plt.ylabel("Nombre de simulations (sur 10)")
plt.ylim(0, 10)
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()



# 1. Courbe moyenne du NeedsIndex par scénario avec écart-type
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=needsindex_df,
    x="Period",
    y="NeedsIndex",
    hue="ScenarioName",
    estimator="mean",
    ci="sd"
)
plt.title("Évolution moyenne du NeedsIndex par scénario (avec écart-type)")
plt.xlabel("Période")
plt.ylabel("NeedsIndex")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Boxplot du NeedsIndex à différentes périodes clés
selected_periods = [5, 10, 15, 20, 24]

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=needsindex_df[needsindex_df["Period"].isin(selected_periods)],
    x="Period",
    y="NeedsIndex",
    hue="ScenarioName"
)
plt.title("Distribution du NeedsIndex à différentes périodes clés")
plt.xlabel("Période")
plt.ylabel("NeedsIndex")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=social_cost_df, x="Period", y="UnemploymentRate", hue="ScenarioName", estimator="mean", ci="sd")
plt.title("Chômage involontaire par scénario (coût social)")
plt.xlabel("Période")
plt.ylabel("Taux de chômage involontaire")
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x="Period", y="BrownCapital", hue="ScenarioName", estimator="mean", ci="sd")
plt.title("Évolution du capital brun moyen par scénario")
plt.xlabel("Période")
plt.ylabel("Capital brun moyen")
plt.grid(True)
plt.tight_layout()
plt.show()

# Style
sns.set(style="whitegrid")

# Fin du chronomètre
execution_time = time.time() - start_time
print(f"Temps d'exécution du modèle : {execution_time:.2f} secondes")

print(f"Taille de all_results : {len(all_results)}")
# Affichage des résultats
results_df = pd.DataFrame(all_results)

# Vérifie combien de firmes sont présentes à chaque période
firmes_par_periode = results_df.groupby("Period")["FirmID"].nunique()

# Affiche un tableau de diagnostic
#print("\nDiagnostic du nombre de firmes par période :")
#for period, count in firmes_par_periode.items():
#    print(f"Période {period} : {count} firmes uniques")

# Affichage de 5 firmes aléatoires sur la dernière période
last_period_df = results_df[results_df["Period"] == num_periods - 1]

plt.figure(figsize=(14, 7))

plt.figure(figsize=(14, 7))

for h, household in enumerate(household_list):
    if "BaseConsumption" in household and "DisposableIncome" in household:
        base = household["BaseConsumption"]
        dispo = household["DisposableIncome"]
        share = [b / d if d > 0.01 else np.nan for b, d in zip(base, dispo)]
        share = [s if s < 2 else np.nan for s in share]  # filtrage des valeurs aberrantes
        plt.plot(range(len(share)), share, alpha=0.3)

plt.xlabel("Périodes")
plt.ylabel("BaseConsumption / DisposableIncome")
plt.title("Part de la base consumption dans le revenu disponible par ménage")
plt.grid(True)
plt.tight_layout()
plt.show()


# Évolution du capital brun pour 5 firmes aléatoires
firm_ids = results_df["FirmID"].drop_duplicates().sample(5, random_state=1).tolist()
subset_df = results_df[(results_df["FirmID"].isin(firm_ids)) & (results_df["Simulation"] == 0)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=subset_df, x="Period", y="BrownCapital", hue="FirmID", legend=True)
plt.title("Évolution du capital brun pour 5 firmes | Simulation 1")
plt.xlabel("Période")
plt.ylabel("BrownCapital")
plt.grid(True)
plt.tight_layout()
plt.show()

# 1. Capital vert moyen par scénario
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x="Period", y="GreenCapital", hue="ScenarioName", estimator="mean", ci="sd")
plt.title("Capital vert moyen par scénario")
plt.tight_layout()
plt.show()



# 2. Revenu disponible moyen par scénario
plt.figure(figsize=(10, 6))
sns.lineplot(data=household_df, x="Period", y="DisposableIncome", hue="ScenarioName", estimator="mean", ci="sd")
plt.title("Revenu disponible moyen par scénario")
plt.tight_layout()
plt.show()



# 4. Boxplot du capital vert à t=24
plt.figure(figsize=(8, 6))
final_period = results_df[results_df["Period"] == 24]
sns.boxplot(data=final_period, x="ScenarioName", y="GreenCapital")
plt.title("Distribution du capital vert à la fin (t=24)")
plt.tight_layout()
plt.show()

"""




# Heatmap du vote individuel par ménage
pivot_votes = vote_df.pivot(index="HouseholdID", columns="Period", values="VoteDecision")

plt.figure(figsize=(14, 10))
sns.heatmap(
    pivot_votes,
    cmap=sns.color_palette(["#d7191c", "#1a9641"], as_cmap=True),  # rouge = anti, vert = pro
    cbar_kws={'label': 'VoteDecision'},
    linewidths=0.1,
    linecolor='grey'
)
plt.title("Heatmap du vote pro-transition par ménage (0 = anti, 1 = pro)")
plt.xlabel("Période")
plt.ylabel("Ménage")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for h_id in pivot_votes.index:
    periods_voted = pivot_votes.columns[pivot_votes.loc[h_id] == 1]
    plt.scatter(periods_voted, [h_id] * len(periods_voted), color="green", s=10, alpha=0.7)

plt.title("Votes pro-transition (1) par ménage et période")
plt.xlabel("Période")
plt.ylabel("ID Ménage")
plt.grid(True)
plt.tight_layout()
plt.show()

vote_prop = vote_df.groupby(["Period", "VoteDecision"]).size().unstack(fill_value=0)
vote_prop = vote_prop[[1, 0]]  # Met le pro-transition en premier
vote_prop_norm = vote_prop.div(vote_prop.sum(axis=1), axis=0)

vote_prop_norm.plot.area(figsize=(10, 6), color=["green", "red"])
plt.title("Évolution de la répartition des votes (pro vs anti-transition)")
plt.xlabel("Période")
plt.ylabel("Proportion")
plt.legend(["Pro-transition (1)", "Anti-transition (0)"])
plt.grid(True)
plt.tight_layout()
plt.show()



plt.figure(figsize=(14, 7))

for h_id in needsindex_df["HouseholdID"].unique():
    household_data = needsindex_df[needsindex_df["HouseholdID"] == h_id]
    plt.plot(household_data["Period"], household_data["NeedsIndex"], alpha=0.3)

plt.title("Évolution du NeedsIndex par ménage (Simulation 0)")
plt.xlabel("Période")
plt.ylabel("NeedsIndex")
plt.grid(True)
plt.tight_layout()
plt.show()




# Évolution du capital vert pour les mêmes firmes
plt.figure(figsize=(12, 6))
sns.lineplot(data=subset_df, x="Period", y="GreenCapital", hue="FirmID", legend=True)
plt.title("Évolution du capital vert pour 5 firmes | Simulation 1")
plt.xlabel("Période")
plt.ylabel("GreenCapital")
plt.grid(True)
plt.tight_layout()
plt.show()

# Boxplot du capital total à 5 périodes sélectionnées
plt.figure(figsize=(10, 6))
selected_periods = [2, 5, 9, 15, 24]
sns.boxplot(x="Period", y="TotalCapital", data=results_df[results_df["Period"].isin(selected_periods)])
plt.title("Distribution du capital total à différentes périodes")
plt.xlabel("Période")
plt.ylabel("TotalCapital")
plt.grid(True)
plt.tight_layout()
plt.show()

# Évolution de la contrainte de crédit pour les 120 firmes
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=results_df[results_df["Simulation"] == 0],
    x="Period",
    y="CreditConstraintVar",
    hue="FirmID",
    legend=False,
    alpha=0.5
)
plt.title("Évolution de la contrainte de crédit pour les 120 firmes | Simulation 1")
plt.xlabel("Période")
plt.ylabel("CreditConstraintVar")
plt.grid(True)
plt.tight_layout()
plt.show()

# 1. Revenu moyen des ménages
plt.figure(figsize=(10, 5))
sns.lineplot(data=household_df, x="Period", y="Income", estimator="mean", ci="sd")
plt.title("Revenu moyen des ménages (Simulation 0)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Consommation moyenne
plt.figure(figsize=(10, 5))
sns.lineplot(data=household_df, x="Period", y="Consumption", estimator="mean", ci="sd")
plt.title("Consommation moyenne des ménages (Simulation 0)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Dette moyenne
plt.figure(figsize=(10, 5))
sns.lineplot(data=household_df, x="Period", y="Debt", estimator="mean", ci="sd")
plt.title("Dette moyenne des ménages (Simulation 0)")
plt.grid(True)
plt.tight_layout()
plt.show() """