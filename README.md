# Hartree-Fock SCF Solver with DIIS

A lightweight Python implementation of the Hartree-Fock Self-Consistent Field (SCF) method. This tool computes the ground-state energy of molecular systems using pre-calculated electronic integrals. It is optimized for small basis sets.

## Usage

The script expects an integral data file as input. By default, it looks for files in a `./data/` directory relative to the script.

### Basic Run
```bash
python hf.py H2.dat
```

### Using Pixi

Pixi is used as the package manager, and a `pixi.toml` is provided. Then one can run 

```bash
pixi run hf
```

with flags appended as desired.

### Command Line Arguments

| Argument | Long Flag | Description | Default |
| :--- | :--- | :--- | :--- |
| input_file | N/A | Path to the integral data file, relative to `./data/` | Required |
| -i | --max_iter | Maximum SCF iterations. | 1000 |
| -e | --e_tol | Energy convergence threshold. | 1e-12 |
| -p | --p_tol | Density matrix convergence threshold. | 1e-10 |
| -m | --mixing | Mixing factor (0.0 = no mixing, 1.0 = full). | 0.5 |
| -d | --disable_diis| Disables DIIS acceleration. | False |
| -H | --diis_history | Number of Fock matrices to store for DIIS. | 6 |
| -t | --diis_tol | Error threshold under which to use DIIS. | 0.1 |
| -u | --units | Converts Enuc from Angstroms to Bohr. | False |

### CUDA and MPI

CUDA and MPI are not supported. Overhead of CUDA and MPI are anticipated to bottleneck performance for the small `n_basis` for which this code is optimized.

## Input File Structure

The parser expects a text file with the following order:
1. Nuclear Repulsion Energy (float)
2. N_basis and N_occ (two integers)
3. Overlap matrix (flattened array)
4. Single-electron Hamiltonian integrals (flattened array)
5. Two-electron integrals (ERIs) (flattened lower-triangle)

## Example data
`H2.dat`, `H2O.dat`, and `LiF.dat` are provided as sample data. The unit conversion behavior is hard-coded for these examples; namely, `H2.dat` and `LiF.dat` require conversion to Bohr. 