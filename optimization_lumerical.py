import numpy as np
import matplotlib.pyplot as plt
from automation_lumerical import *

def calculate_kappa_shg(Electric_fields):
    Electric_field_fundamental = Electric_fields[0]
    Electric_field_shg = Electric_fields[1]

    y = Electric_field_fundamental['y'].squeeze()   # (Ny,)
    z = Electric_field_fundamental['z'].squeeze()   # (Nz,)
    Electric_field_y_fundamental = Electric_field_fundamental['E'][0, :, :, 0, 1] 
    Electric_field_y_shg = Electric_field_shg['E'][0, :, :, 0, 1] 

    I = Electric_field_y_fundamental**2*np.conj(Electric_field_y_shg)    # intengrand

    int_z  = np.trapezoid(I, z, axis=1)       # resultado depende de y
    numerator = np.trapezoid(int_z, y, axis=0)   # escalar: ∫∫ |E|^2 dy dz
    Pf_z = np.trapezoid(np.abs(Electric_field_y_fundamental)**2, z, axis=1)
    Pf = np.trapezoid(Pf_z, y)

    P2_z = np.trapezoid(np.abs(Electric_field_y_shg)**2, z, axis=1)
    P2 = np.trapezoid(P2_z, y)

    denominator = np.sqrt(Pf**2 * P2)
    kappa = np.abs(numerator) / denominator

    return kappa

def confinement_factor(E, core_mask):
    Ex = E['E'][0,:,:,0,0]
    Ey = E['E'][0,:,:,0,1]
    Ez = E['E'][0,:,:,0,2]

    E2 = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    return np.sum(E2 * core_mask) / np.sum(E2)

def make_core_mask(E_field, h, w):
    y = E_field['y']   # Lumerical coordinate arrays
    z = E_field['z']

    Y, Z = np.meshgrid(y, z, indexing='ij')

    mask = (np.abs(Y) <= w/2) & (np.abs(Z) <= h/2)
    return mask.astype(float)

def objective(mode, h, w, L_core, y_span, n_trial_modes, angle, wavelengths, weights=None, metric='prod'):
    confinement = [0,0]
    mode, cfg = build(mode, h, w, L_core, y_span, n_trial_modes, angle)
    Electric_fields, n_effs, indices = solve_wavelengths(mode, cfg, wavelengths)
    mask0 = make_core_mask(Electric_fields[0], h, w-2*h/np.tan(np.deg2rad(angle)))
    mask1 = make_core_mask(Electric_fields[1], h, w-2*h/np.tan(np.deg2rad(angle)))

    kappa = calculate_kappa_shg(Electric_fields)
    confinement[0] = confinement_factor(Electric_fields[0], mask0)
    confinement[1] = confinement_factor(Electric_fields[1], mask1)

    if metric == "sum":
        val= weights["weight_kappa"]*1e13*kappa + weights["weight_E0"]*confinement[0]+weights["weight_E1"]*confinement[1]
    if metric == "prod":
        val= kappa*confinement[0]*confinement[1]

    return np.abs(val), np.abs(kappa), np.array(confinement)

import numpy as np
import os


# =========================================================
# CHECKPOINT SAVE
# =========================================================
def save_checkpoint(filename, step, h, w, best_h, best_w, best_val, best_kappa, best_confinement,
                    hs, ws, vals, kappas, confinements, T):
    np.savez(
        filename,
        step=step,
        h=h,
        w=w,
        best_h=best_h,
        best_w=best_w,
        best_val=best_val,
        best_kappa=best_kappa,
        best_confinement=best_confinement,
        hs=hs,
        ws=ws,
        vals=vals,
        kappas=kappas,
        confinements=confinements,
        T=T,
    )
    print(f"💾 Saved checkpoint at step {step} -> {filename}")


# =========================================================
# CHECKPOINT LOAD
# =========================================================
def load_checkpoint(filename):
    if not os.path.exists(filename):
        print("⚠ No checkpoint found, starting fresh.")
        return None

    data = np.load(filename, allow_pickle=True)
    print(f"🔄 Loaded checkpoint from {filename}")

    return {
        "step": int(data["step"]),
        "h": float(data["h"]),
        "w": float(data["w"]),
        "best_h": float(data["best_h"]),
        "best_w": float(data["best_w"]),
        "best_val": float(data["best_val"]),
        "best_kappa": float(data["best_kappa"]),
        "best_confinement":list(data["best_confinement"]),
        "hs": list(data["hs"]),
        "ws": list(data["ws"]),
        "kappas": list(data["kappas"]),
        "T": float(data["T"]),
    }


# =========================================================
# ANNEAL WITH SAVE + RESUME
# =========================================================
def anneal(mode,
           h0, w0,
           h_bounds, w_bounds,
           L_core, y_span, n_trial_modes, angle, wavelengths,
           n_steps=100, T0=3, alpha=0.95, step_size=0.1, weights={"weight_kappa":0.3,"weight_E0":0.3,"weight_E1":0.4},
           checkpoint_file="anneal_checkpoint.npz",
           resume=False):

    # -----------------------------------------------------
    # Load from checkpoint if requested
    # -----------------------------------------------------
    if resume:
        ckpt = load_checkpoint(checkpoint_file)
        if ckpt is not None:
            start_step = ckpt["step"] + 1
            h = ckpt["h"]
            w = ckpt["w"]
            best_h = ckpt["best_h"]
            best_w = ckpt["best_w"]
            best_val = ckpt["best_val"]
            best_kappa = ckpt["best_kappa"]
            best_confinement = ckpt["best_confinement"]
            hs = ckpt["hs"]
            ws = ckpt["ws"]
            kappas = ckpt["kappas"]
            T = ckpt["T"]
        else:
            # No checkpoint found – start fresh
            start_step = 0
            h, w = h0, w0
            best_h, best_w = h, w
            best_val, best_kappa, best_confinement = objective(mode, h, w, L_core, y_span,
                                   n_trial_modes, angle, wavelengths, weights)
            hs, ws, vals, kappas, confinements = [], [], [], [], []
            T = T0
    else:
        # Start fresh explicitly
        start_step = 0
        h, w = h0, w0
        best_h, best_w = h, w
        best_val, best_kappa, best_confinement = objective(mode, h, w, L_core, y_span,
                               n_trial_modes, angle, wavelengths, weights)
        hs, ws, vals, kappas, confinements = [], [], [], [], []
        T = T0

    # -----------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------
    for k in range(start_step, n_steps):

        # propose new sample
        h_new = np.clip(h * (1 + step_size*np.random.randn()), *h_bounds)
        w_new = np.clip(w * (1 + step_size*np.random.randn()), *w_bounds)

        # evaluate
        val_new, kappa_new, confinement_new = objective(mode, h_new, w_new,
                            L_core, y_span, n_trial_modes, angle, wavelengths, weights)

        # record series
        hs.append(best_h)
        ws.append(best_w)
        vals.append(best_val)
        kappas.append(best_kappa)
        confinements.append(best_confinement)

        # Metropolis/annealing condition
        dE = val_new - best_val
        if dE > 0 or np.random.rand() < np.exp(dE / T):
            h, w = h_new, w_new
            if val_new > best_val:
                best_val = val_new
                best_h, best_w = h_new, w_new
                best_kappa, best_confinement = kappa_new, confinement_new


        # cool down
        T *= alpha

        print(f"step {k}, T={T:.3f}, best |kappa|={best_kappa:.3e}")

        # ---------------------------------------------
        # Save checkpoint every step (or change interval)
        # ---------------------------------------------
        save_checkpoint(
            checkpoint_file,
            step=k,
            h=h,
            w=w,
            best_h=best_h,
            best_w=best_w,
            best_val=best_val,
            best_kappa=best_kappa,
            best_confinement=best_confinement,
            hs=np.array(hs),
            ws=np.array(ws),
            vals=np.array(vals),
            kappas=np.array(kappas),
            confinements=np.array(confinements),
            T=T,
        )

    return best_h, best_w, best_val, best_kappa, best_confinement, np.array(hs), np.array(ws), np.array(vals), np.array(kappas), np.array(confinements)
