# SECRYPT 2025
## Can contributing more put you at a higher leakage risk? The Relationship between Shapley Value and training data leakage risks in Federated Learning
### Repository Information
This repository accompanies the paper "Can contributing more put you at a higher leakage risk? The Relationship between Shapley Value and training data leakage risks in Federated Learning?", accepted for presentation at SECRYPT 2025.

It contains:
  - The code used to generate all experimental results,
  - The full set of numerical results, and
  - A Jupyter Notebook with the analysis and visualization of the results.
Please refer to the paper for a detailed description of the methods and experimental setup.

### Repository Structure
The following repository is structured as follows:

 - ./SHAP_MIA contains all the relevant code used for performing the simulation.
 - ./attack_results contains all the numerical results obtained during the simulation. This concerns either the results of the Federated simulation or the results of the attacks.
 - ./experiments contains code used for performing simulation runs (the basic blocks are imported from ./SHAP_MIA).
 - ./tables contains all the generated .tex table containing numerical results (correlation, cross-correlation or stationarity tests).
 - ./tests contains all the unit test assessing the code used for generating the experiments.

Additionally:

  -./visualization.ipynb contains all the code for generating figures and tables.
  -./pyproject.toml allows for using Poetry environment to install the project's dependencies.

  ### Numerical Results Retrieval
  - To retrieve the numerical results of Federated simulation (SHAP or LOO values, accuracy and precission or other registered metrics), navigate into the ./attack_results/{split}/{with_or_without_DP}/{dataset}/results.
  - To retrieve the numerical results of MIA, navigate into the ./attack_results/{split}/{with_or_without_DP}/{dataset}/nodes_attack_results.json or -||-/orchestrator_attack_results.json.
  - To retrieve the TeX tables containing numerical results regarding correlation, cross-correlation or stationarity, navigate into ./tables.
  - To retrieve the code for generating either the tables or the figures, navigate into the ./visualizations.ipynb.
