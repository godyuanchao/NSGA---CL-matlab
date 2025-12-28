function run_sensitivity_analysis()
    % Sensitivity Analysis Script - Response to Reviewer 6 regarding hyperparameter robustness
    % Testing the impact of different weight configurations on algorithm performance
    
    clc; clear; close all;
    
    % Experiment settings
    n_runs = 10; % Run 10 times per configuration (30 times recommended for formal paper)
    
    % Define weight configurations (w1=Rank, w2=Quality, w3=Diversity, w4=Gradient)
    configs = {
        'Proposed (0.4/0.3/0.2/0.1)', [0.4, 0.3, 0.2, 0.1];
        'Equal Weights',              [0.25, 0.25, 0.25, 0.25];
        'Rank Dominant',              [0.7, 0.1, 0.1, 0.1];
        'CL Dominant',                [0.1, 0.3, 0.3, 0.3];
    };
    
    num_configs = size(configs, 1);
    
    fprintf('=== Start Sensitivity Analysis ===\n');
    fprintf('Total %d weight configurations, running %d times each\n\n', num_configs, n_runs);
    
    % Result storage
    results = cell(num_configs, 4); % Name, Spacing, Conv, HV
    
    for c = 1:num_configs
        config_name = configs{c, 1};
        weights = configs{c, 2};
        
        % Inject weights into Base Workspace for tournamentsel_cl.m to read
        assignin('base', 'CL_WEIGHTS', weights);
        
        fprintf('Testing configuration: %s ...\n', config_name);
        fprintf('    Weights: Rank=%.2f, Qual=%.2f, Div=%.2f, Grad=%.2f\n', ...
            weights(1), weights(2), weights(3), weights(4));
        
        spacing_vals = zeros(n_runs, 1);
        conv_vals = zeros(n_runs, 1);
        hv_vals = zeros(n_runs, 1);
        
        for r = 1:n_runs
            % Set random seed to r*2 for consistency
            current_seed = r * 2;
            
            % Always run Full NSGA-II-CL mode ([1, 1, 1])
            [sp, cv, hv] = run_single_experiment(current_seed, 1, 1, 1);
            
            spacing_vals(r) = sp;
            conv_vals(r) = cv;
            hv_vals(r) = hv;
        end
        
        % Calculate results
        fprintf('  Spacing: %.4f +/- %.4f\n', mean(spacing_vals), std(spacing_vals));
        fprintf('  Conv:    %.4f +/- %.4f\n', mean(conv_vals), std(conv_vals));
        fprintf('  HV:      %.4f +/- %.4f\n', mean(hv_vals), std(hv_vals));
        fprintf('------------------------------------------------\n');
        
        % Format result string "Mean +/- Std"
        results{c, 1} = config_name;
        results{c, 2} = sprintf('%.4f +/- %.4f', mean(spacing_vals), std(spacing_vals));
        results{c, 3} = sprintf('%.4f +/- %.4f', mean(conv_vals), std(conv_vals));
        results{c, 4} = sprintf('%.4f +/- %.4f', mean(hv_vals), std(hv_vals));
    end
    
    % Clear Base Workspace variables
    evalin('base', 'clear CL_WEIGHTS');
    
    % Output results to CSV
    T = cell2table(results, 'VariableNames', {'Weight_Configuration', 'Spacing', 'Convergence', 'HV'});
    writetable(T, 'sensitivity_analysis_results.csv');
    fprintf('\nSensitivity analysis results saved to sensitivity_analysis_results.csv\n');
end

function [final_spacing, final_convergence, final_hv] = run_single_experiment(seed, use_cross_cl, use_mut_cl, use_sel_cl)
    % Single experiment logic, integrating logic from main.m and main_cl.m
    rng(seed);
    
    % === Parameter Settings ===
    nvar = 3;
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
    
    % Contrastive learning parameters
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
    
    if use_any_cl
        cl_data = contrastive_learning(pop, F, cl_temperature, cl_alpha);
    else
        cl_data = [];
    end
    
    % === Main Loop ===
    for it = 1:maxit
        popc = repmat(empty, nc/2, 2);
        
        for j = 1:nc/2
            % 1. Selection
            if use_sel_cl
                % Use CL selection (tournamentsel_cl will read external CL_WEIGHTS)
                p1 = tournamentsel_cl(pop, cl_data);
                p2 = tournamentsel_cl(pop, cl_data);
            else
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
        
        % Update CL parameters
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
            [pop, F] = nondominatedsort(pop);
            pop = calcrowdingdistance(pop, F);
        end
        
        pop = Sortpop(pop);
    end
    
    % === Calculate Final Metrics ===
    F1 = pop(F{1});
    if ~isempty(F1)
        costs = [F1.cost]';
        f_min = min(costs, [], 1);
        f_max = max(costs, [], 1);
        range_f = f_max - f_min;
        range_f(range_f == 0) = 1;
        costs_norm = (costs - f_min) ./ range_f;
        
        if size(costs_norm, 1) > 1
            final_spacing = compute_spacing(costs_norm);
        else
            final_spacing = 0;
        end
        final_convergence = mean(sqrt(sum(costs_norm.^2, 2)));
        ref_norm = [1.1, 1.1];
        final_hv = hv2d_min(costs_norm, ref_norm);
    else
        final_spacing = 0;
        final_convergence = inf;
        final_hv = 0;
    end
end

% === Helper Functions ===
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
