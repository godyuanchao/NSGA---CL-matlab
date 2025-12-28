function run_ablation_experiment()
    % Ablation Experiment Script - Response to Reviewer 6 regarding parameter and module effectiveness
    % Run 5 variants, 30 runs each, calculate Mean +/- Std
    
    clc; clear; close all;
    
    % Experiment settings
    n_runs = 30; % Number of runs per variant
    variants = {
        'Baseline (NSGA-II)',      [0, 0, 0]; % [Use_Cross_CL, Use_Mut_CL, Use_Sel_CL]
        '+ Sim Crossover',         [1, 0, 0];
        '+ Grad Mutation',         [0, 1, 0];
        '+ Selection Score',       [0, 0, 1];
        'Full NSGA-II-CL',         [1, 1, 1];
    };
    
    num_variants = size(variants, 1);
    
    fprintf('=== Start Ablation Study ===\n');
    fprintf('Total %d variants, %d runs per variant\n\n', num_variants, n_runs);
    
    % Result storage (Variant + 3 metrics Mean +/- Std strings)
    summary_results = cell(num_variants, 4);
    
    % Pre-allocate Raw Data storage (Variant, RunID, Seed, Spacing, Conv, HV)
    raw_data = cell(num_variants * n_runs, 6);
    raw_idx = 1;

    for v = 1:num_variants
        variant_name = variants{v, 1};
        flags = variants{v, 2};
        use_cross_cl = flags(1);
        use_mut_cl = flags(2);
        use_sel_cl = flags(3);
        
        fprintf('Running variant: %s ...\n', variant_name);
        
        spacing_vals = zeros(n_runs, 1);
        conv_vals = zeros(n_runs, 1);
        hv_vals = zeros(n_runs, 1);
        
        for r = 1:n_runs % Use serial calculation for debugging
            % Set random seed to r*2 to match compare_algorithms.m seed sequence
            current_seed = r * 2;
            fprintf('    Run %d/%d (Seed %d)...\n', r, n_runs, current_seed);
            [sp, cv, hv] = run_single_experiment(current_seed, use_cross_cl, use_mut_cl, use_sel_cl);
            spacing_vals(r) = sp;
            conv_vals(r) = cv;
            hv_vals(r) = hv;
            
            % Store Raw Data
            raw_data{raw_idx, 1} = variant_name;
            raw_data{raw_idx, 2} = r;
            raw_data{raw_idx, 3} = current_seed;
            raw_data{raw_idx, 4} = sp;
            raw_data{raw_idx, 5} = cv;
            raw_data{raw_idx, 6} = hv;
            raw_idx = raw_idx + 1;
        end
        
        % Calculate results and print to console (Matlab Command Window)
        fprintf('  Done. \n');
        fprintf('  Spacing: %.4f +/- %.4f\n', mean(spacing_vals), std(spacing_vals));
        fprintf('  Conv:    %.4f +/- %.4f\n', mean(conv_vals), std(conv_vals));
        fprintf('  HV:      %.4f +/- %.4f\n', mean(hv_vals), std(hv_vals));
        fprintf('------------------------------------------------\n');
        
        summary_results{v, 1} = variant_name;
        summary_results{v, 2} = sprintf('%.4f +/- %.4f', mean(spacing_vals), std(spacing_vals));
        summary_results{v, 3} = sprintf('%.4f +/- %.4f', mean(conv_vals), std(conv_vals));
        summary_results{v, 4} = sprintf('%.4f +/- %.4f', mean(hv_vals), std(hv_vals));
    end
    
    % Output Summary Results to CSV
    T_summary = cell2table(summary_results, 'VariableNames', {'Variant', 'Spacing', 'Convergence', 'HV'});
    writetable(T_summary, 'ablation_study_results.csv');
    fprintf('Ablation study summary results saved to ablation_study_results.csv\n');
    
    % Output Raw Data to CSV
    T_raw = cell2table(raw_data, 'VariableNames', {'Variant', 'Run_ID', 'Seed', 'Spacing', 'Convergence', 'HV'});
    writetable(T_raw, 'ablation_study_raw_data.csv');
    fprintf('\nAblation study raw data saved to ablation_study_raw_data.csv\n');
end

function [final_spacing, final_convergence, final_hv] = run_single_experiment(seed, use_cross_cl, use_mut_cl, use_sel_cl)
    % Single experiment logic, integrating logic from main.m and main_cl.m
    rng(seed);
    
    % === Parameter Settings ===
    nvar = 3;
    nobj = 2;
    npop = 50;
    maxit = 100;
    pc = 0.8;
    nc = round(pc * npop / 2) * 2;
    mu = 0.05;
    
    varmin = [1.5, 40, 30];
    varmax = [2.5, 50, 75];
    step = [0.01, 1, 1];
    len = (varmax - varmin) ./ step;
    var = [varmin; step; varmax; round(len, 0)];
    
    % Contrastive learning parameters (used only when CL features are required)
    use_any_cl = use_cross_cl || use_mut_cl || use_sel_cl;
    
    cl_temperature = 0.1;
    cl_alpha = 0.5;
    cl_update_freq = 5;
    
    % === Initialization ===
    empty.position = [];
    empty.cost = [];
    empty.rank = [];
    empty.domination = [];
    empty.dominated = 0;
    empty.crowdingdistance = [];
    pop = repmat(empty, npop, 1);
    
    for i = 1:npop
        pop(i).position = create_x(var);
        pop(i).cost = costfunction(pop(i).position);
    end
    
    [pop, F] = nondominatedsort(pop);
    pop = calcrowdingdistance(pop, F);
    
    % If any CL component is enabled, calculate CL data
    if use_any_cl
        cl_data = contrastive_learning(pop, F, cl_temperature, cl_alpha);
    else
        cl_data = []; % Placeholder
    end
    
    % === Main Loop ===
    for it = 1:maxit
        popc = repmat(empty, nc/2, 2);
        
        for j = 1:nc/2
            % 1. Selection
            if use_sel_cl
                % Use CL selection
                p1 = tournamentsel_cl(pop, cl_data);
                p2 = tournamentsel_cl(pop, cl_data);
            else
                % Use traditional selection
                p1 = tournamentsel(pop);
                p2 = tournamentsel(pop);
            end
            
            % 2. Crossover
            if use_cross_cl
                p1_idx = find_individual_index(pop, p1);
                p2_idx = find_individual_index(pop, p2);
                [popc(j, 1).position, popc(j, 2).position] = ...
                    crossover_cl(p1.position, p2.position, cl_data, p1_idx, p2_idx, var);
            else
                % Use traditional crossover
                [popc(j, 1).position, popc(j, 2).position] = crossover(p1.position, p2.position);
            end
        end
        popc = popc(:);
        
        for k = 1:nc
            % 3. Mutation
            if use_mut_cl
                similar_idx = find_most_similar_individual(popc(k).position, pop, cl_data);
                popc(k).position = mutate_cl(popc(k).position, mu, var, cl_data, similar_idx);
            else
                % Use traditional mutation
                popc(k).position = mutate(popc(k).position, mu, var);
            end
            
            popc(k).cost = costfunction(popc(k).position);
        end
        
        % Merge and Environmental Selection
        newpop = [pop; popc];
        [newpop, F_new] = nondominatedsort(newpop);
        newpop = calcrowdingdistance(newpop, F_new);
        newpop = Sortpop(newpop);
        pop = newpop(1:npop);
        
        % Update CL parameters (only when using CL)
        if use_any_cl
            if mod(it, cl_update_freq) == 0 || it == 1
                [pop, F] = nondominatedsort(pop);
                pop = calcrowdingdistance(pop, F);
                cl_data = contrastive_learning(pop, F, cl_temperature, cl_alpha);
                
                [cl_temperature, cl_alpha, mu] = adaptive_parameter_update(it, maxit, ...
                    cl_temperature, cl_alpha, mu, cl_data.contrastive_loss);
            else
                [pop, F] = nondominatedsort(pop);
                pop = calcrowdingdistance(pop, F);
            end
        else
            % Traditional mode, only need to recalculate rank and crowding distance
            [pop, F] = nondominatedsort(pop);
            pop = calcrowdingdistance(pop, F);
        end
        
        pop = Sortpop(pop);
    end
    
    % === Calculate Final Metrics ===
    F1 = pop(F{1});
    if ~isempty(F1)
        costs = [F1.cost]';
        
        % Normalization calculation
        f_min = min(costs, [], 1);
        f_max = max(costs, [], 1);
        range_f = f_max - f_min;
        range_f(range_f == 0) = 1;
        costs_norm = (costs - f_min) ./ range_f;
        
        % Spacing
        if size(costs_norm, 1) > 1
            final_spacing = compute_spacing(costs_norm);
        else
            final_spacing = 0;
        end
        
        % Convergence
        final_convergence = mean(sqrt(sum(costs_norm.^2, 2)));
        
        % HV
        ref_norm = [1.1, 1.1];
        final_hv = hv2d_min(costs_norm, ref_norm);
    else
        final_spacing = 0;
        final_convergence = inf;
        final_hv = 0;
    end
end

% === Helper Functions (Copied from main_cl.m) ===
function idx = find_individual_index(pop, individual)
    for i = 1:length(pop)
        if isequal(pop(i).position, individual.position)
            idx = i;
            return;
        end
    end
    idx = 1;
end

function similar_idx = find_most_similar_individual(position, pop, cl_data)
    min_distance = inf;
    similar_idx = 1;
    for i = 1:length(pop)
        distance = norm(position - pop(i).position);
        if distance < min_distance
            min_distance = distance;
            similar_idx = i;
        end
    end
end

function [temp, alpha, mu] = adaptive_parameter_update(iteration, max_iteration, ...
    current_temp, current_alpha, current_mu, contrastive_loss)
    progress = iteration / max_iteration;
    temp = current_temp * (1 - 0.5 * progress);
    temp = max(temp, 0.01);
    
    if progress < 0.5
        alpha = current_alpha * (1 + 0.2 * progress);
    else
        alpha = current_alpha * (1.1 - 0.2 * progress);
    end
    alpha = max(alpha, 0.1);
    alpha = min(alpha, 0.8);
    
    if contrastive_loss > 1.0
        mu = current_mu * 1.1;
    elseif contrastive_loss < 0.3
        mu = current_mu * 0.95;
    else
        mu = current_mu;
    end
    mu = max(mu, 0.03);
    mu = min(mu, 0.15);
end

function spacing = compute_spacing(costs)
    if size(costs, 1) < 2
        spacing = 0;
        return;
    end
    n = size(costs, 1);
    distances = zeros(n, 1);
    for i = 1:n
        min_dist = inf;
        for j = 1:n
            if i ~= j
                dist = norm(costs(i,:) - costs(j,:));
                if dist < min_dist
                    min_dist = dist;
                end
            end
        end
        distances(i) = min_dist;
    end
    mean_dist = mean(distances);
    spacing = sqrt(sum((distances - mean_dist).^2) / n);
end

function hv = hv2d_min(costs, ref)
    if isempty(costs)
        hv = 0;
        return
    end
    C = sortrows(costs, [1 2]);
    hv = 0;
    prev_x = ref(1);
    for i = size(C,1):-1:1
        x = C(i,1);
        y = C(i,2);
        w = prev_x - x;
        h = ref(2) - y;
        if w > 0 && h > 0
            hv = hv + w*h;
            prev_x = x;
        end
    end
end