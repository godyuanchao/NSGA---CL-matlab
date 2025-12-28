 %% Algorithm comparison verification script
% This script performs main comparison (statistical analysis) and auxiliary comparison (relative gain)
% According to the modification strategy:
% 1. Main comparison: Report metrics mean +/- std under the same budget and 30 random seeds
% 2. Auxiliary comparison: Calculate Normalized Relative Gain

clc;
clear;
close all;
num_runs = 30; % Number of runs
is_subroutine = true; % Flag as subroutine mode to avoid main script clearing variables

% Initialize result storage
metrics_base = struct('spacing', [], 'convergence', [], 'hv', []);
metrics_cl = struct('spacing', [], 'convergence', [], 'hv', []);
fronts_base_raw = {};
fronts_cl_raw = {};

fprintf('Starting algorithm comparison experiment (Total %d runs)...\n', num_runs);

for run_idx = 1:num_runs
   
    current_seed = run_idx * 2 ;
    
    %% 1. Run traditional NSGA-II
    rng(current_seed); % Set seed
    fprintf('Run %d (Seed %d): Running NSGA-II ...\n', run_idx, current_seed);
    try
        run('main.m');
        metrics_base.spacing(end+1) = final_spacing;
        metrics_base.convergence(end+1) = final_convergence;
        metrics_base.hv(end+1) = final_hv;
        
        if exist('costsF', 'var')
            fronts_base_raw{end+1} = costsF;
        else
            fronts_base_raw{end+1} = [];
        end
    catch ME
        fprintf('NSGA-II Error: %s\n', ME.message);
        fronts_base_raw{end+1} = [];
    end
    
    %% 2. Run improved algorithm NSGA-II-CL
    rng(current_seed); % Set same seed for fair comparison
    fprintf('Run %d (Seed %d): Running NSGA-II-CL ...\n', run_idx, current_seed);
    try
        run('main_cl.m');
        metrics_cl.spacing(end+1) = final_spacing;
        metrics_cl.convergence(end+1) = final_convergence;
        metrics_cl.hv(end+1) = final_hv;
        
        if exist('costsF', 'var')
            fronts_cl_raw{end+1} = costsF;
        else
            fronts_cl_raw{end+1} = [];
        end
    catch ME
        fprintf('NSGA-II-CL Error: %s\n', ME.message);
        fronts_cl_raw{end+1} = [];
    end
    
    if run_idx == num_runs
        fprintf('Iteration metrics saved to nsga2_cl_metrics.csv\n');
        fprintf('Pareto front data saved to nsga2_cl_pareto_front.csv\n');
    end
end

%% Calculate statistical results (Mean +/- Std) - Main Comparison
metric_names = {'spacing', 'convergence', 'hv'};
metric_labels = {'SPACING', 'Convergence', 'HV'};
% Metric direction: 1 means bigger is better, -1 means smaller is better
metric_directions = [-1, -1, 1]; 

results_base = struct();
results_cl = struct();

fprintf('\n\n=================================================================================\n');
fprintf('                                Main Comparison (Mean +/- Std)                               \n');
fprintf('=================================================================================\n');
fprintf('%-15s | %-25s | %-25s | %-10s\n', 'Metric', 'NSGA-II (Baseline)', 'NSGA-II-CL (Proposed)', 'Result');
fprintf('---------------------------------------------------------------------------------\n');

for i = 1:length(metric_names)
    name = metric_names{i};
    label = metric_labels{i};
    direction = metric_directions(i);
    
    % Baseline
    data_b = metrics_base.(name);
    mu_b = mean(data_b);
    std_b = std(data_b);
    results_base.(name).mu = mu_b;
    
    % Proposed
    data_cl = metrics_cl.(name);
    mu_cl = mean(data_cl);
    std_cl = std(data_cl);
    results_cl.(name).mu = mu_cl;
    
    % Determine winner
    if direction == 1 % Bigger is better
        if mu_cl > mu_b
            res = 'Win';
        else
            res = 'Lose';
        end
    else % Smaller is better
        if mu_cl < mu_b
            res = 'Win';
        else
            res = 'Lose';
        end
    end
    
    fprintf('%-15s | %10.4f +/- %-10.4f | %10.4f +/- %-10.4f | %-10s\n', ...
        label, mu_b, std_b, mu_cl, std_cl, res);
end

%% Calculate relative gain
fprintf('\n\n=================================================================================\n');
fprintf('                                Relative Gain                          \n');
fprintf('=================================================================================\n');
fprintf('%-15s | %-20s\n', 'Metric', 'Improvement (%)');
fprintf('---------------------------------------------------------------------------------\n');

for i = 1:length(metric_names)
    name = metric_names{i};
    label = metric_labels{i};
    direction = metric_directions(i);
    
    mu_b = results_base.(name).mu;
    mu_cl = results_cl.(name).mu;
    
    if direction == 1 % Bigger is better
        gain = (mu_cl - mu_b) / abs(mu_b) * 100;
    else % Smaller is better
        gain = (mu_b - mu_cl) / abs(mu_b) * 100;
    end
    
    fprintf('%-15s | %+.2f%%\n', label, gain);
end
fprintf('=================================================================================\n');

%% Save raw data to CSV
raw_data_table = table((1:num_runs)', ...
    metrics_base.spacing(:), metrics_base.convergence(:), metrics_base.hv(:), ...
    metrics_cl.spacing(:), metrics_cl.convergence(:), metrics_cl.hv(:), ...
    'VariableNames', {'Run_ID', 'Base_Spacing', 'Base_Convergence', 'Base_HV', ...
                      'CL_Spacing', 'CL_Convergence', 'CL_HV'});

csv_filename = 'algorithm_comparison_raw_data.csv';
writetable(raw_data_table, csv_filename);


