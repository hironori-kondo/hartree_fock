import argparse
import numpy as np
from scipy import linalg
from numba import njit

"""
This code implements a basic Hartree-Fock SCF procedure with optional DIIS acceleration.
It accepts command-line arguments for various parameters and reads integral data from a specified file.
The code is optimized for small basis sets.

The following functions are deprecated.
They have been superseded by more optimized implementations.
Note that performance limits are unusual due to small n_basis.

@njit
def _unpack_eri(eri_flat, n_basis):
    eri = np.empty((n_basis, n_basis, n_basis, n_basis), dtype=eri_flat.dtype)
    idx = 0  

    # Iterate over cases where rs <= pq    
    for p in range(n_basis):
        for q in range(p + 1):

            # Step 1: Process cases where r < p.
            # This gurantees rs < pq.
            for r in range(p):
                for s in range(r + 1):
                    _assign_8fold(eri, p, q, r, s, eri_flat[idx])
                    idx += 1
            
            # Step 2: Process cases where r==p.
            # Then we need s <= q to maintain rs <= pq.
            r = p
            for s in range(q + 1):
                _assign_8fold(eri, p, q, r, s, eri_flat[idx])
                idx += 1
    return eri

@njit(inline='always')
def _assign_8fold(eri, p, q, r, s, val):
    for i, j in ((p, q), (q, p)):
        for k, l in ((r, s), (s, r)):
            eri[i, j, k, l] = val
            eri[k, l, i, j] = val

def scf_loop_diis(enuc, n_basis, n_occ, S, h, eri, max_iter=100, e_tol=1e-8, p_tol=1e-6, diis_history = 6, diis_tol = 0.1):
    s_evals, s_evecs = linalg.eigh(S)
    X = s_evecs @ np.diag(s_evals**(-0.5)) @ s_evecs.T
    P_old = np.zeros((n_basis, n_basis), dtype=np.float64)
    energy_old = 0.0

    # DIIS Storage: Storing larger lists for speed
    Fs = np.zeros((diis_history, n_basis, n_basis), dtype=np.float64)
    errs = np.zeros((diis_history, n_basis, n_basis), dtype=np.float64)
    err_ortho = diis_tol + 1.0  # Initialize above threshold to skip DIIS on first iteration
    F_count = 0

    for iteration in range(1, max_iter + 1):
        F_raw = _build_fock(h, eri, P_old)
        F_raw = (F_raw + F_raw.T) / 2.0
        energy_scf = np.sum(P_old * (h + F_raw))
        
        # Calculate DIIS error vector using commutaor of Fock and density matrices
        if iteration > 1:
            err_ortho = X.T @ (F_raw @ P_old @ S - S @ P_old @ F_raw) @ X

        # Perform DIIS extrapolation
        F = F_raw
        if np.max(np.abs(err_ortho)) < diis_tol:
            current_index = F_count % diis_history
            Fs[current_index, :, :] = F_raw
            errs[current_index, :, :] = err_ortho
            F_count += 1

            if F_count >  1: # Need at least 2 Fock matrices to perform extrapolation
                F_dim = np.minimum(F_count, diis_history)
                B_dim = F_dim + 1
                B = np.zeros((B_dim, B_dim))
                B[:-1, :-1] = np.einsum('iab, jab -> ij', errs[:F_dim], errs[:F_dim], optimize=True)
                B_max = np.max(np.abs(B[:-1, :-1]))
                if B_max > 0:
                    B[:-1, :-1] /= B_max

                B[-1, :] = -1
                B[:, -1] = -1
                B[-1, -1] = 0
                
                b = np.zeros(B_dim)
                b[-1] = -1
                
                try:
                    c = np.linalg.pinv(B) @ b
                    F = np.einsum('i, ijk -> jk', c[:-1], Fs[:F_dim], optimize=True)
                except np.linalg.LinAlgError:
                    pass 
        
        # 5. Get New Density from the extrapolated Fock matrix
        P, epsilon = _get_density(F, X, n_occ)
        
        delta_e = np.abs(energy_scf - energy_old)
        delta_p = np.linalg.norm(P - P_old)
        print(f"Iteration {iteration}: E_scf={energy_scf:.10f}, ΔE={delta_e:.2e}, ΔP={delta_p:.2e}")
        
        if delta_e < e_tol and delta_p < p_tol:
            energy_total = energy_scf + enuc
            print(f"SCF converged: E_total ={energy_total:.10f}, ΔE={delta_e:.2e}, ΔP={delta_p:.2e}")
            break
        
        energy_old = energy_scf
        P_old = P
    else:
        print("SCF did not converge within the maximum iterations.")
    
    return P, energy_total, epsilon

def scf_loop(enuc, n_basis, n_occ, S, h, eri, max_iter=1000, e_tol = 1e-12, p_tol=1e-10, mixing = 0.5):
    s_evals, s_evecs = linalg.eigh(S)
    X = s_evecs @ np.diag(s_evals**(-0.5)) @ s_evecs.T
    P_old = np.zeros((n_basis, n_basis), dtype=np.float64)
    energy_old = 0.0

    for iteration in range(1, max_iter + 1):
        P_raw, energy_scf, epsilon = _scf_iteration(n_basis, n_occ, h, eri, P_old, X)
        
        if iteration > 1:
            P = (1.0 - mixing) * P_raw + mixing * P_old
        else:
            P = P_raw

        delta_e = np.abs(energy_scf - energy_old)
        delta_p = np.linalg.norm(P - P_old)
        print(f"Iteration {iteration}: E_scf={energy_scf:.10f}, ΔE={delta_e:.2e}, ΔP={delta_p:.2e}")
        
        if delta_e < e_tol and delta_p < p_tol:
            energy_total = energy_scf + enuc
            print(f"SCF converged: E_total ={energy_total:.10f}, ΔE={delta_e:.2e}, ΔP={delta_p:.2e}")
            break
        
        energy_old = energy_scf
        P_old = P
    else:
        print("SCF did not converge within the maximum number of iterations.")
    
    return P, energy_total, epsilon

def _scf_iteration(n_basis, n_occ, h, eri, P, X):
    F = _build_fock(h, eri, P)
    F = (F + F.T) / 2.0
    P_new, epsilon = _get_density(F, X, n_occ)
    energy = np.sum(P * (h + F)) # Omit factor of 0.5 here
    return P_new, energy, epsilon

def _build_fock(h, eri, P):
    J = np.einsum('rs,pqrs->pq', P, eri, optimize=True)
    K = np.einsum('rs,prqs->pq', P, eri, optimize=True)
    F = h + 2*J - K
    return F
"""
def parse_file(filename, convert_units=True):
    """Reads integral daata from a file and unpacks it into the necessary components for Hartree-Fock."""
    with open(filename, 'r') as f:
        try:
            enuc = np.float64(next(f).strip()) # Line 1: Nuclear repulsion energy.
            if convert_units:
                enuc /= 1.889726  # Units wrong (\A -> Bohr needed)
            n_basis, n_occ = map(np.int64, next(f).split()) # Line 2: Number of basis functions and occupied orbitals
            print(f"Parsed Data: Enuc={enuc}, N_basis={n_basis}, N_occ={n_occ}")
            
            # Line 3 and 4: Overlap and Hamiltonian matrices
            s_flat = np.fromstring(next(f), sep=' ', dtype=np.float64)
            h_flat = np.fromstring(next(f), sep=' ', dtype=np.float64)
            s = s_flat.reshape((n_basis, n_basis))
            h = h_flat.reshape((n_basis, n_basis))
            print(f"Overlap Matrix S shape: {s.shape}, Sample value S[0,0]={s[0,0]}")
            print(f"Hamiltonian Matrix H shape: {h.shape}, Sample value H[0,0]={h[0,0]}")

            # Line 5: ERIs (N.B.: Only the lower triangle is stored due to symmetry)
            eri_flat = np.fromstring(next(f), sep=' ', dtype=np.float64)
            eri = _unpack_eri(eri_flat, n_basis)
            print(f"ERI Tensor shape: {eri.shape}, Sample value eri[0,0,0,0]={eri[0,0,0,0]}")

            return enuc, n_basis, n_occ, s, h, eri

        except Exception as e:
            print(f"Error parsing file: {e}")
            raise e

def _unpack_eri(eri_flat, n_basis):    
    M = n_basis * (n_basis + 1) // 2
    eri_2d = np.zeros((M, M), dtype=np.float64)
    tril_idx = np.tril_indices(M)
    eri_2d[tril_idx] = eri_flat
    
    # Symmetrize to handle (pq <-> rs) symmetry
    eri_2d = eri_2d + eri_2d.T - np.diag(eri_2d.diagonal())
    
    # Get the mapping from the compound index back to (p, q) pairs
    p_idx, q_idx = np.tril_indices(n_basis)
    
    # Break 2D indices into 4D indices
    I, J = np.meshgrid(np.arange(M), np.arange(M), indexing='ij')
    p1, q1 = p_idx[I], q_idx[I]
    p2, q2 = p_idx[J], q_idx[J]
    
    # Assign the remaining 4-fold symmetries (p <-> q and r <-> s). The meshgrid implicitly handles the I <-> J swap.
    eri = np.zeros((n_basis, n_basis, n_basis, n_basis), dtype=np.float64)
    eri[p1, q1, p2, q2] = eri_2d
    eri[q1, p1, p2, q2] = eri_2d
    eri[p1, q1, q2, p2] = eri_2d
    eri[q1, p1, q2, p2] = eri_2d
    
    return eri

def scf_loop(enuc, n_basis, n_occ, S, h, eri, max_iter=1000, e_tol = 1e-12, p_tol=1e-10, mixing = 0.5, use_diis = True, diis_history = 6, diis_tol = 0.1):
    s_evals, s_evecs = linalg.eigh(S)
    X = s_evecs @ np.diag(s_evals**(-0.5)) @ s_evecs.T
    P_old = np.zeros((n_basis, n_basis), dtype=np.float64)
    energy_old = 0.0

    if use_diis:
        Fs = np.zeros((diis_history, n_basis, n_basis), dtype=np.float64)
        errs = np.zeros((diis_history, n_basis, n_basis), dtype=np.float64)
        B_full = np.zeros((diis_history + 1, diis_history + 1), dtype=np.float64)
        F_count = 0

    n_sq = n_basis ** 2
    eri_J = eri.reshape(n_sq, n_sq) # J: 'rs, pqrs -> pq'
    eri_K = np.transpose(eri, (0, 2, 1, 3)).copy().reshape(n_sq, n_sq) # K: 'rs, prqs -> pq'

    for iteration in range(1, max_iter + 1):
        P_flat = P_old.flatten()
        J = (eri_J @ P_flat).reshape(n_basis, n_basis)
        K = (eri_K @ P_flat).reshape(n_basis, n_basis)
        F_raw = h + 2*J - K
        F_raw = (F_raw + F_raw.T) / 2.0
        energy_scf = np.sum(P_old * (h + F_raw)) # Omit factor of 0.5 here
        diis_exec = False
        F = F_raw

        if iteration > 1:
            if use_diis:
                PS = P_old @ S
                err_ortho = X.T @ (F_raw @ PS - PS.T @ F_raw) @ X

                if np.max(np.abs(err_ortho)) < diis_tol:
                    current_index = F_count % diis_history
                    Fs[current_index, :, :] = F_raw
                    errs[current_index, :, :] = err_ortho
                    F_count += 1

                    if F_count > 1:
                        F_dim = np.minimum(F_count, diis_history)
                        B_dim = F_dim + 1
                        B = B_full[:B_dim, :B_dim]

                        err_flat = errs[:F_dim].reshape(F_dim, -1)
                        B[:-1, :-1] = err_flat @ err_flat.T

                        B_max = np.max(np.abs(B[:-1, :-1]))
                        if B_max > 0:
                            B[:-1, :-1] /= B_max

                        B[-1, :-1] = -1
                        B[:-1, -1] = -1
                        np.fill_diagonal(B, B.diagonal() + 1e-12)
                        B[-1, -1] = 0.0
                        
                        b = np.zeros(B_dim)
                        b[-1] = -1
                        
                        try:
                            c = linalg.solve(B, b, assume_a='sym')
                            F = (c[:-1] @ Fs[:F_dim].reshape(F_dim, -1)).reshape(n_basis, n_basis)
                            diis_exec = True
                        except linalg.LinAlgError:
                            pass 
            if not diis_exec:
                F = (1.0 - mixing) * F_raw + mixing * F_old

        F_ortho = X.T @ F @ X
        epsilon, C_ortho = linalg.eigh(F_ortho, lower=True, check_finite=False)
        C = X @ C_ortho
        C_occ = C[:, :n_occ]
        P = np.dot(C_occ, C_occ.T) # Omit factor of 2 here

        delta_e = np.abs(energy_scf - energy_old)
        delta_p = np.sqrt(np.sum((P - P_old)**2))
        energy_total = energy_scf + enuc
        diis_flag = " (DIIS)" if diis_exec else ""
        print(f"Iteration {iteration}{diis_flag}: E_total={energy_total:.10f}, ΔE={delta_e:.2e}, ΔP={delta_p:.2e}")

        if delta_e < e_tol and delta_p < p_tol:
            print(f"SCF converged: E_total ={energy_total:.10f}, ΔE={delta_e:.2e}, ΔP={delta_p:.2e}")
            break
        
        energy_old = energy_scf
        P_old = P
        F_old = F
    else:
        print("SCF did not converge within the maximum number of iterations.")

    return P, energy_total, epsilon

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Hartree-Fock SCF calculation from integral data file.")
    parser.add_argument("input_file", help="Path to the integral data.")
    parser.add_argument("-i", "--max_iter", type=int, default=1000, help="Maximum number of SCF iterations.")
    parser.add_argument("-e", "--e_tol", type=float, default=1e-12, help="Energy convergence threshold.")
    parser.add_argument("-p", "--p_tol", type=float, default=1e-10, help="Density matrix convergence threshold.")
    parser.add_argument("-m", "--mixing", type=float, default=0.5, help="Fock matrix mixing parameter (0-1). Value of 0 means no mixing, 1 means full mixing.")
    parser.add_argument("-d", "--disable_diis", action="store_true", help="Disable DIIS acceleration.")
    parser.add_argument("-H", "--diis_history", type=int, default=6, help="Number of Fock matrices to store for DIIS.")
    parser.add_argument("-t", "--diis_tol", type=float, default=0.1, help="DIIS error threshold for accepting Fock matrices into the DIIS extrapolation.")
    parser.add_argument("-u", "--units", action="store_true", help="If specified, nuclear repulsion energy will be converted to Hartree units assuming distances are in angstroms.")
    args = parser.parse_args()

    filename = f"./data/{args.input_file}"

    conversion_map = {"H2.dat": True, "LiF.dat": True}
    enuc, n_basis, n_occ, S, h, eri = parse_file(filename, conversion_map.get(filename, False) or args.units)

    scf_loop(enuc, n_basis, n_occ, S, h, eri, max_iter=args.max_iter, e_tol=args.e_tol, p_tol=args.p_tol, mixing=args.mixing, use_diis=not args.disable_diis, diis_history=args.diis_history, diis_tol=args.diis_tol)