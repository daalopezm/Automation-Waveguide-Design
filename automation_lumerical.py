import lumapi
import numpy as np
import os


# ---------------------------------------------------------
# CONFIGURATION (constants that rarely change)
# ---------------------------------------------------------
def get_base_config():
    """Return default constants EXCEPT the geometric parameters that are passed explicitly."""
    return {
        "t_LN": 0.6e-6,
        "t_BOX": 2.0e-6,
        "t_CLAD": 1.0e-6,
        "t_Si": 1.0e-6,
        "z_center": 0,
        "base_angle": 60,
        "mat_Si": "Si (Silicon) - Palik",
        "mat_BOX": "SiO2 (Glass) - Palik",
        "mat_LN": "LN_X_cut",
        "mat_CLAD": "SiO2 (Glass) - Palik",
    }


# ---------------------------------------------------------
# MATERIAL DB
# ---------------------------------------------------------
def load_material_db(mode, mdf_file):
    mdf_path = os.path.abspath(mdf_file)
    mode.eval(f'importmaterialdb("{mdf_path}");')
    print("✔ Material DB loaded:", mdf_path)


# ---------------------------------------------------------
# MESH
# ---------------------------------------------------------
def add_mesh(mode, cfg):
    mode.addmesh()
    mode.set("name", "mesh")
    mode.set("override x mesh", 1)
    mode.set("override y mesh", 1)
    mode.set("override z mesh", 1)
    mode.set("dx", 20e-9)
    mode.set("dy", 20e-9)
    mode.set("dz", 20e-9)
    mode.set("x span", cfg["L_core"])
    mode.set("y span", cfg["y_span"])
    mode.set("z span", cfg["t_LN"] + cfg["t_CLAD"] + cfg["t_BOX"])


# ---------------------------------------------------------
# GEOMETRY FUNCTIONS
# ---------------------------------------------------------

def add_ln_film(mode, cfg):
    mode.addrect()
    mode.set("name", "LN_base")
    substrate_left=cfg["t_LN"]-cfg["h_core"]
    cfg.update({"substrate_left":substrate_left})
    mode.set("material", cfg["mat_LN"])
    mode.set("x span", cfg["L_core"])
    mode.set("y span", cfg["y_span"])
    mode.set("z span", cfg["substrate_left"])

    z = cfg["z_center"] - cfg["h_core"]/2 - cfg["substrate_left"]/2
    mode.set("z", z)

def add_silicon(mode, cfg):
    mode.addrect()
    mode.set("name", "Silicon")
    mode.set("material", cfg["mat_Si"])
    mode.set("x span", cfg["L_core"])
    mode.set("y span", cfg["y_span"])
    mode.set("z span", cfg["t_Si"])

    h_core = cfg["h_core"]
    z = cfg["z_center"] - (2*cfg["t_BOX"] + 2*cfg["substrate_left"] + h_core)/2 - cfg["t_Si"]/2
    mode.set("z", z)


def add_box(mode, cfg):
    mode.addrect()
    mode.set("name", "BOX")
    mode.set("material", cfg["mat_BOX"])
    mode.set("x span", cfg["L_core"])
    mode.set("y span", cfg["y_span"])
    mode.set("z span", cfg["t_BOX"])

    h_core = cfg["h_core"]
    z = cfg["z_center"] - (2*cfg["substrate_left"] + h_core)/2 - cfg["t_BOX"]/2
    mode.set("z", z)


def add_ln_ridge(mode, cfg):
    mode.addobject("straight_wg")
    mode.set("name", "straight_wg")
    mode.set("z", cfg["z_center"])
    mode.set("material", cfg["mat_LN"])
    mode.set("base height", cfg["h_core"])
    mode.set("base width", cfg["w_core"])
    mode.set("base angle", cfg["base_angle"])
    mode.set("x span", cfg["L_core"])
    print(f"thickness LN {cfg["h_core"]+ cfg["substrate_left"]}")



def add_cladding(mode, cfg):
    h_core = cfg["h_core"]
    w_core = cfg["w_core"]
    y_span = cfg["y_span"]
    angle = np.deg2rad(cfg["base_angle"])
    t_CLAD = cfg["t_CLAD"]
    zc = cfg["z_center"]

    y = np.array([
        -y_span/2,
        -w_core/2,
        -w_core/2 + h_core/np.tan(angle),
        w_core/2 - h_core/np.tan(angle),
        w_core/2,
        y_span/2,
        y_span/2,
        -y_span/2
    ])

    z = np.array([
        zc - h_core/2,
        zc - h_core/2,
        h_core - h_core/2,
        h_core - h_core/2,
        zc - h_core/2,
        zc - h_core/2,
        t_CLAD - h_core/2,
        t_CLAD - h_core/2
    ])

    vertices = np.vstack([y, z])

    mode.addpoly()
    mode.set("name", "Cladding_poly")
    mode.set("material", cfg["mat_CLAD"])
    mode.set("vertices", vertices)
    mode.set("z span", cfg["L_core"])
    mode.set("first axis", "z")
    mode.set("rotation 1", 90)
    mode.set("second axis", "y")
    mode.set("rotation 2", 90)


# ---------------------------------------------------------
# MODE SOLVER
# ---------------------------------------------------------
def setup_and_run_fde(mode, cfg):
    mode.set("wavelength", cfg["lambda0"])
    mode.set("number of trial modes", cfg["n_trial_modes"])
    mode.findmodes()


# ---------------------------------------------------------
# MODE EXTRACTION
# ---------------------------------------------------------
def extract_te_modes(mode, n_trial_modes):
    E_fields = []
    neffs = []
    for i in range(n_trial_modes):
        pf = mode.getresult(f"FDE::data::mode{i+1}", "TE polarization fraction")
        if pf > 0.5:
            E = mode.getresult(f"FDE::data::mode{i+1}", "E")
            neff = mode.getresult(f"FDE::data::mode{i+1}", "neff")
            E_fields.append(E)
            neffs.append(neff)
    mode.switchtolayout()
    return E_fields, neffs


# ---------------------------------------------------------
# FIND BEST MODE
# ---------------------------------------------------------
def find_strongest_ey_mode(E_fields):
    max_val = -np.inf
    max_i = None

    for i, data in enumerate(E_fields):
        Ey = data["E"][0, :, :, 0, 1]
        cx, cy = Ey.shape[0]//2, Ey.shape[1]//2
        val = np.abs(Ey[cx, cy])
        if val > max_val:
            max_val = val
            max_i = i

    return max_i, max_val


# ---------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------
def initialize_lumerical_MODE(mdf_file="xcut_zcut_linbo3.mdf"):
    mode = lumapi.MODE(hide=False)
    load_material_db(mode, mdf_file)
    return mode

def build(
    mode,
    h_core,
    w_core,
    L_core,
    y_span,
    n_trial_modes,
    angle
):
    """Main function to build geometry, solve modes, and return best TE mode index."""
    
    cfg = get_base_config()
    cfg.update({
        "h_core": h_core,
        "w_core": w_core,
        "L_core": L_core,
        "y_span": y_span,
        "n_trial_modes": n_trial_modes,
        "base_angle": angle
    })

    
    mode.deleteall()
    add_mesh(mode, cfg)
    add_ln_film(mode, cfg)
    add_box(mode, cfg)
    add_silicon(mode, cfg)   
    add_ln_ridge(mode, cfg)
    add_cladding(mode, cfg)
    mode.addfde()
    mode.set("solver type", 1)
    mode.set("y span", cfg["y_span"])
    mode.set("z", 0)
    mode.set("z span", cfg["h_core"] + cfg["t_CLAD"])

    return mode, cfg



# =========================================================
# 2) SOLVE FOR ONE WAVELENGTH
# =========================================================
def solve_for_wavelength(mode, cfg, wavelength):
    """
    Runs FDE, extracts TE modes, picks the mode with strongest Ey.
    Returns (best_mode_field, neff, mode_index)
    """
    cfg["lambda0"] = wavelength
    setup_and_run_fde(mode, cfg)
    E_fields, neffs = extract_te_modes(mode, cfg["n_trial_modes"])

    idx, val = find_strongest_ey_mode(E_fields)

    print(f"\n=== λ = {wavelength*1e9:.1f} nm ===")
    print("✔ Strongest Ey mode index:", idx)
    print("✔ Max |Ey| at center:", val)
    print("✔ n_eff:", neffs[idx])

    return E_fields[idx], neffs[idx], idx


# =========================================================
# 3) SOLVE FOR MULTIPLE WAVELENGTHS
# =========================================================
def solve_wavelengths(mode, cfg, wavelength_list):
    fields = []
    neffs = []
    idx_list = []

    for wl in wavelength_list:
        E, n_eff, idx = solve_for_wavelength(mode, cfg, wl)
        fields.append(E)
        neffs.append(n_eff)
        idx_list.append(idx)

    return fields, neffs, idx_list
