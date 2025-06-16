# Kinetic Modelling of CO₂ Capture and Utilisation on NiRu–Ca/Al Dual-Function Material (DFM)

This repository contains a transparent and fully documented implementation of the kinetic parameter estimation framework for **CO₂ capture and methanation** using a novel **NiRu–Ca/Al dual-function material (DFM)**. The modelling framework simulates the **cyclic operation** of a fixed-bed DFM reactor across **adsorption**, **purge**, and **hydrogenation** stages, with experimental validation and open-source reproducibility in mind.

---

## Overview

**Dual-function materials (DFMs)** integrate CO₂ adsorption and catalytic methanation in a single solid-phase material, enabling process intensification and improved energy efficiency in Power-to-Gas (PtG) systems. This study focuses on a newly synthesised 15 wt% Ni, 1 wt% Ru, 10 wt% CaO/Al₂O₃ DFM, characterised experimentally and modelled mechanistically.

The kinetic model simulates reactor behaviour using:
- 1D finite difference method (FDM)
- Coupled mass balances (gas & surface)
- Implicit solver with backward Euler scheme
- System delay correction via 2nd-order response model
- Parameter estimation using Bayesian optimisation (`Optuna`)

---

## Repository Contents

DFM_Kinetics_PE/
│
├── Adsorption_PE.py # Parameter estimation for the CO₂ adsorption stage
├── Purge_Hydrogenation_PE.py # Combined purge + hydrogenation parameter estimation
├── data/ # Experimental time-resolved concentration profiles
│ ├── Adsorption_Data_delagged.xlsx
│ ├── CH4-380.xlsx
│ ├── CH4-300.xlsx
│ ├── CH4-220.xlsx
│ ├── H2O-380.xlsx
│ ├── H2O-300.xlsx
│ ├── H2O-220.xlsx
│ ├── CO2-380.xlsx
│ ├── CO2-300.xlsx
│ └── CO2-220.xlsx

yaml
Copy
Edit

---

## Requirements

Install the required Python packages:

```bash
pip install numpy pandas matplotlib scipy optuna
Tested with Python 3.10+ in a conda environment.

How to Run
Each script is standalone and can be run directly to perform simulation and parameter fitting:

1. Adsorption Stage:
bash
Copy
Edit
python Adsorption_PE.py
This will:

Simulate CO₂ and H₂O adsorption profiles

Apply a second-order delay filter

Fit kinetic constants (k₁, k₂, k₃) using Optuna

Visualise CO₂ and H₂O breakthrough curves

2. Purge + Hydrogenation Stages:
bash
Copy
Edit
python Purge_Hydrogenation_PE.py
This script:

Links purge and hydrogenation stages

Propagates coverage factors across stages

Optimises parameters across three temperature sets (220°C, 300°C, 380°C)

Fits CH₄, H₂O, and CO₂ outlet profiles to experimental data

Experimental Background
Experiments were performed on a fixed-bed reactor using a NiRu–Ca/Al DFM, capturing CO₂ from a 12.2% CO₂/N₂ stream and converting it to CH₄ using 10% H₂/N₂. Time-resolved FT-IR gas analysis was performed at three temperatures (493 K, 573 K, and 653 K) across three stages: adsorption (20 min), purge (15 min), and hydrogenation (30 min).

Details of the experimental setup, synthesis method, and validation protocol are available in the associated manuscript (currently under review). The DOI will be added upon publication.

Kinetic Framework Highlights
Mechanistic models developed for:

CO₂ and H₂O adsorption on CaO-based sorbent

Carbonate decomposition during purge

Sabatier methanation using approach-to-equilibrium kinetics

Surface coverages (θ_CO₂, θ_H₂O) tracked across stages

Analyser delay incorporated using 2nd-order dynamic response model

Optuna used for robust global optimisation (TPE sampler)

Model validated at multiple temperatures

Citation
Please cite this work as:

Dolat, M., Wright, A., Bahrami Gharamaleki, S., Merkouri, L.P., Duyar, M.S., & Short, M.
Kinetic modelling of the CO₂ capture and utilisation on NiRu–Ca/Al dual function material via parameter estimation, 2025.
DOI: (to be updated upon publication)

Contact
For questions or collaboration inquiries, please contact:
Meshkat Dolat – [m.dolat@surrey.ac.uk]
Dr. Michael Short – [m.short@surrey.ac.uk]

License
This repository is released under the Creative Commons Attribution (CC BY 4.0) license.

Acknowledgements
This work was supported by the University of Surrey's School of Chemistry and Chemical Engineering, and funded in part by the EPSRC [grant EP/X000753/1].
