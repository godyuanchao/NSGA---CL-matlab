# NSGA-II-CL: Contrastive Learning Enhanced Multi-Objective Evolutionary Algorithm

This repository contains the source code for the paper "Enhancing Multi-Objective Evolutionary Algorithms with Contrastive Learning". It includes the implementation of the proposed NSGA-II-CL algorithm, the traditional NSGA-II baseline, and scripts for algorithm comparison, ablation studies, and sensitivity analysis.

## Repository Structure

### Core Algorithms
- **`main.m`**: Implementation of the baseline NSGA-II algorithm.
- **`main_cl.m`**: Implementation of the proposed NSGA-II-CL (Contrastive Learning enhanced NSGA-II) algorithm.

### Experiments & Analysis
- **`compare_algorithms.m`**: Main comparison script. Runs both algorithms 30 times with fixed random seeds to ensure fair comparison and generates statistical results (Mean +/- Std).
- **`run_ablation_experiment.m`**: Performs ablation studies to validate the effectiveness of specific modules (Crossover, Mutation, Selection).
- **`run_sensitivity_analysis.m`**: Conducts sensitivity analysis on hyperparameter configurations.

### Helper Functions
- **`tournamentsel_cl.m`**: Contrastive learning enhanced tournament selection operator.
- **`contrastive_learning.m`**: Core logic for contrastive learning features and loss calculation.
- **`nondominatedsort.m`**, **`calcrowdingdistance.m`**: Standard NSGA-II operators.
- **`costfunction.m`**: The multi-objective optimization problem definition.

## Reproducibility

To ensure reproducibility as requested by the peer review process, we provide full details on the experimental setup, parameter settings, and random seed management.

### System Requirements
- MATLAB R2021a or later.
- No additional toolboxes are strictly required, but the Statistics and Machine Learning Toolbox is recommended for some analysis functions.

### Parameter Settings
To ensure a fair comparison, both the baseline NSGA-II and the proposed NSGA-II-CL share the same basic evolutionary parameters.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `npop` | 50 | Population Size |
| `maxit` | 100 | Maximum Generations |
| `pc` | 0.8 | Crossover Probability |
| `mu` | 0.05 | Mutation Probability |
| `nvar` | 3 | Number of Design Variables |
| `nobj` | 2 | Number of Objectives |

**NSGA-II-CL Specific Parameters:**
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `cl_temperature` | 0.1 | Temperature parameter for contrastive loss |
| `cl_alpha` | 0.5 | Learning rate / Weight factor |
| `cl_update_freq` | 5 | Frequency of contrastive model updates |

### Random Seeds
- **Comparison Experiments**: The `compare_algorithms.m` script explicitly sets the random seed for each run to ensure that both algorithms start from the same initial population and face the same stochastic conditions.
    - Formula: `current_seed = run_idx * 2` (where `run_idx` goes from 1 to 30).
- **Ablation & Sensitivity**: These scripts also use fixed seeds for each variant/configuration to guarantee consistent results across different runs.

## Usage Instructions

### 1. Run Algorithm Comparison
To reproduce the main comparison results (Table 1 & Figure 5/6 in the manuscript):
```matlab
run('compare_algorithms.m')
```
*Outputs:*
- `nsga2_metrics.csv` / `nsga2_cl_metrics.csv`: Iteration-wise metrics.
- `comparison_results.mat`: Statistical summary.
- Console output showing Mean +/- Std for Spacing, Convergence, and HV.

### 2. Run Ablation Study
To reproduce the ablation experiment results:
```matlab
run('run_ablation_experiment.m')
```
*Outputs:*
- `ablation_study_results.csv`: Summary of performance for different algorithm variants.

### 3. Run Sensitivity Analysis
To reproduce the sensitivity analysis results:
```matlab
run('run_sensitivity_analysis.m')
```
*Outputs:*
- `sensitivity_analysis_results.csv`: Performance metrics under different weight configurations.

## Contact
For any questions regarding the code or the paper, please contact the authors via email.
