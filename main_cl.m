if ~exist('is_subroutine', 'var')
    clc;
    clear;
    close all;
    rng(40);
end
%% Algorithm parameter settings
nvar = 3;           % Number of design variables
nobj = 2;           % Number of objective functions
npop = 50;          % Population size
maxit = 100;        % Maximum iteration
pc = 0.8;           % Crossover probability
nc = round(pc * npop / 2) * 2;  % Number of crossover individuals
mu = 0.05;          % Mutation probability

% Example Problem: Design variable constraints
 varmin = [1.5,40,30];% Lower bound
 varmax = [2.5,50,75];% Upper bound
 step = [0.01, 1, 1];% Step size
len = (varmax - varmin) ./ step;
var = [varmin; step; varmax; round(len, 0)];

% Contrastive learning parameters
cl_temperature = 0.1;    % Temperature parameter
cl_alpha = 0.5;          % Learning rate
cl_update_freq = 5;      % Contrastive learning update frequency

%% Initialize data structures
empty.position = [];
empty.cost = [];
empty.rank = [];
empty.domination = [];
empty.dominated = 0;
empty.crowdingdistance = [];
pop = repmat(empty, npop, 1);

% Performance record
performance_history = struct();
performance_history.best_costs = [];
performance_history.convergence_generation = [];
performance_history.spacing = [];
performance_history.convergence = [];
performance_history.hv = [];
performance_history.diversity = [];
performance_history.contrastive_loss = [];
performance_history.hv_final = [];
%% Initialize population
for i = 1:npop
    pop(i).position = create_x(var);
    pop(i).cost = costfunction(pop(i).position);
end

%% Initial non-dominated sorting and crowding distance calculation
[pop, F] = nondominatedsort(pop);
pop = calcrowdingdistance(pop, F);

% Initialize contrastive learning
cl_data = contrastive_learning(pop, F, cl_temperature, cl_alpha);

if ~exist('is_subroutine', 'var') || ~is_subroutine
    fprintf('Initialization complete, starting evolution process...\n\n');
end

%% Main evolution loop
for it = 1:maxit
    if ~exist('is_subroutine', 'var') || ~is_subroutine
        fprintf('Evolution generation %d...\n', it);
    end
    
    %% Generate offspring population
    popc = repmat(empty, nc/2, 2);
    
    % Use contrastive learning enhanced selection and crossover
    for j = 1:nc/2
        % Use enhanced selection operator
        p1 = tournamentsel_cl(pop, cl_data);
        p2 = tournamentsel_cl(pop, cl_data);
        
        % Find index of parent individual in population
        p1_idx = find_individual_index(pop, p1);
        p2_idx = find_individual_index(pop, p2);
        
        % Use contrastive learning enhanced crossover operator
        [popc(j, 1).position, popc(j, 2).position] = ...
            crossover_cl(p1.position, p2.position, cl_data, p1_idx, p2_idx, var);
    end
    
    popc = popc(:);
    
    % Use contrastive learning enhanced mutation
    for k = 1:nc
        % Find the most similar individual index in the original population (for mutation guidance)
        similar_idx = find_most_similar_individual(popc(k).position, pop, cl_data);
        
        popc(k).position = mutate_cl(popc(k).position, mu, var, cl_data, similar_idx);
        popc(k).cost = costfunction(popc(k).position);
    end
    
    %% Merge parents and offspring
    newpop = [pop; popc];
    
    %% Non-dominated sorting and crowding distance calculation
    [newpop, F_new] = nondominatedsort(newpop);
    newpop = calcrowdingdistance(newpop, F_new);
    
    % Sort
    newpop = Sortpop(newpop);
    
    % Environmental selection
    pop = newpop(1:npop);
    
    %% Update contrastive learning information
    if mod(it, cl_update_freq) == 0 || it == 1
        [pop, F] = nondominatedsort(pop);
        pop = calcrowdingdistance(pop, F);
        
        % Update contrastive learning data
        cl_data = contrastive_learning(pop, F, cl_temperature, cl_alpha);
        
        % Adaptively adjust parameters
        [cl_temperature, cl_alpha, mu] = adaptive_parameter_update(it, maxit, ...
            cl_temperature, cl_alpha, mu, cl_data.contrastive_loss);
    else
        [pop, F] = nondominatedsort(pop);
        pop = calcrowdingdistance(pop, F);
    end
    
    pop = Sortpop(pop);
    
    %% Record performance metrics
    if ~isempty(F) && ~isempty(F{1})
        F1 = pop(F{1});
        performance_history = update_performance_history(performance_history, F1, pop, cl_data, it);
    else
        % If no non-dominated solutions, create empty performance record
    performance_history.contrastive_loss(it) = cl_data.contrastive_loss;
    performance_history.convergence_generation(it) = it;
    performance_history.spacing(it) = 0;
    F1 = [];
end
    
    %% Display evolution information

    
    % Plotting - Only keep the final figure6 output
end

%% Result analysis and output
F1_final = pop(F{1});

%% Performance metric output

% Calculate final performance metrics
final_convergence_generation = performance_history.convergence_generation(end);
final_spacing = performance_history.spacing(end);
final_contrastive_loss = performance_history.contrastive_loss(end);

% Calculate convergence metric (average objective function value)
if ~isempty(F1_final)
    costs = [F1_final.cost]';
    final_convergence = mean(sqrt(sum(costs.^2, 2))); % Average Euclidean distance
else
    final_convergence = inf;
end

% Calculate metric improvement
if length(performance_history.spacing) > 1
    spacing_improvement = (performance_history.spacing(1) - final_spacing) / max(performance_history.spacing(1), 1e-10) * 100; % SPACING: smaller is better
    
    % Calculate convergence improvement rate (using initial convergence metric from history)
    if length(performance_history.convergence) > 1
        initial_convergence = performance_history.convergence(1);
        convergence_improvement = (initial_convergence - final_convergence) / max(initial_convergence, 1e-10) * 100;
    else
        convergence_improvement = 0;
    end
end
if ~isempty(F1_final)
    costsF = [F1_final.cost]';
    if size(costsF, 2) ~= 2
         % If cost is originally a row vector, [F1_final.cost] will become 1x(2N) or Nx2 depending on specific structure, but here cost is known to be 2x1 column vector
         % [F1.cost] -> 2xN. Transpose -> Nx2. Correct.
    end
    % HV normalization calculation
    % 1. Get min and max of Pareto front as normalization bounds
    f_min = min(costsF, [], 1);
    f_max = max(costsF, [], 1);
    
    % 2. Prevent division by zero if max equals min
    range_f = f_max - f_min;
    range_f(range_f == 0) = 1; 
    
    % 3. Normalize to [0, 1]
    costs_norm = (costsF - f_min) ./ range_f;
    
    % 4. Set reference point to [1.1, 1.1] (slightly larger than 1 to include boundary points)
    ref_norm = [1.1, 1.1];
    
    % 5. Calculate normalized HV
    final_hv = hv2d_min(costs_norm, ref_norm);
    
    % 7. Calculate normalized SPACING
    final_spacing = compute_spacing(costs_norm);
    
    % 8. Calculate normalized convergence (distance to ideal point [0,0])
    final_convergence = mean(sqrt(sum(costs_norm.^2, 2)));

    % 9. Calculate normalized diversity (removed)
    
    else
        final_hv = 0;
        final_spacing = 0;
        final_convergence = inf;
    end
    performance_history.hv_final = final_hv;
    if ~exist('is_subroutine', 'var') || ~is_subroutine
        fprintf('Final Normalized Metrics:\n');
        fprintf('SPACING: %.6f\n', final_spacing);
        fprintf('Convergence: %.6f\n', final_convergence);
        fprintf('HV: %.6f\n', final_hv);
    end
    
    % Ensure variables are available in base workspace for analysis scripts
    assignin('base', 'final_spacing', final_spacing);
    assignin('base', 'final_convergence', final_convergence);
    assignin('base', 'final_hv', final_hv);
    if exist('costsF', 'var')
        assignin('base', 'costsF', costsF);
    end

    if ~isempty(performance_history.spacing)
    init_spacing = performance_history.spacing(1);
    ir_spacing = (init_spacing - final_spacing) / max(init_spacing, 1e-10) * 100;
end
if ~isempty(performance_history.convergence)
    init_conv = performance_history.convergence(1);
    ir_conv = (init_conv - final_convergence) / max(init_conv, 1e-10) * 100;
end
if evalin('base','exist(''baseline_final_spacing'',''var'')')
    b_spacing = evalin('base','baseline_final_spacing');
    gain_spacing = (b_spacing - final_spacing) / max(b_spacing, 1e-10) * 100;
end
if evalin('base','exist(''baseline_final_convergence'',''var'')')
    b_conv = evalin('base','baseline_final_convergence');
    gain_conv = (b_conv - final_convergence) / max(b_conv, 1e-10) * 100;
end
if evalin('base','exist(''baseline_final_hv'',''var'')')
    b_hv = evalin('base','baseline_final_hv');
    gain_hv = (final_hv - b_hv) / max(b_hv, 1e-10) * 100;
end



% Optimal solution info saved to result file

% Save results
if evalin('base','exist(''results_filename'',''var'')')
    rf = evalin('base','results_filename');
    save(rf, 'pop', 'F', 'performance_history', 'cl_data');
else
    save('nsga2_cl_results.mat', 'pop', 'F', 'performance_history', 'cl_data');
end

% Save Spacing, Convergence, HV of each generation to CSV
metrics_table = table((1:length(performance_history.spacing))', ...
                      performance_history.spacing(:), ...
                      performance_history.convergence(:), ...
                      performance_history.hv(:), ...
                      'VariableNames', {'Generation', 'Spacing', 'Convergence', 'HV'});
writetable(metrics_table, 'nsga2_cl_metrics.csv');

% Output figure6 numerical data (sorted by Objective_1 ascending)
costs = [F1_final.cost];

% Extract data and transform Objective_1 values
obj1_values = -costs(1, :);
obj2_values = costs(2, :);

% Sort by Objective_1 values in ascending order
[sorted_obj1, sorted_indices] = sort(obj1_values);
sorted_obj2 = obj2_values(sorted_indices);


% Save Pareto front data to CSV
pareto_table = table(sorted_obj1(:), sorted_obj2(:), ...
                     'VariableNames', {'Objective_1', 'Objective_2'});
writetable(pareto_table, 'nsga2_cl_pareto_front.csv');
if ~exist('is_subroutine', 'var') || ~is_subroutine
    fprintf('Iteration metrics saved to nsga2_cl_metrics.csv\n');
    fprintf('Pareto front data saved to nsga2_cl_pareto_front.csv\n');
end

%% Helper functions
function idx = find_individual_index(pop, individual)
    % Find index of individual in population
    for i = 1:length(pop)
        if isequal(pop(i).position, individual.position)
            idx = i;
            return;
        end
    end
    idx = 1;  % Default return first
end

function similar_idx = find_most_similar_individual(position, pop, cl_data)
    % Find index of individual most similar to given position
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
    % Adaptive parameter update
    progress = iteration / max_iteration;
    
    % Temperature parameter: gradually decrease with evolution
    temp = current_temp * (1 - 0.5 * progress);
    temp = max(temp, 0.01);  % Minimum value limit
    
    % Learning rate: higher in middle stage, lower in later stage
    if progress < 0.5
        alpha = current_alpha * (1 + 0.2 * progress);
    else
        alpha = current_alpha * (1.1 - 0.2 * progress);
    end
    alpha = max(alpha, 0.1);
    alpha = min(alpha, 0.8);
    
    % Mutation probability: adaptively adjusted based on contrastive loss
    if contrastive_loss > 1.0  % Increase mutation when loss is high
        mu = current_mu * 1.1;
    elseif contrastive_loss < 0.3  % Decrease mutation when loss is low
        mu = current_mu * 0.95;
    else
        mu = current_mu;
    end
    mu = max(mu, 0.03);
    mu = min(mu, 0.15);
end

function performance_history = update_performance_history(performance_history, F1, pop, cl_data, iteration)
    % Update performance history record
    
    % Record contrastive loss
    performance_history.contrastive_loss(iteration) = cl_data.contrastive_loss;
    
    % Record convergence generation
    performance_history.convergence_generation(iteration) = iteration;
    
    % Calculate SPACING metric
    if ~isempty(F1) && isfield(F1, 'cost') && ~isempty([F1.cost])
        costs = [F1.cost]';
        
        % Normalize costs first
        f_min = min(costs, [], 1);
        f_max = max(costs, [], 1);
        range_f = f_max - f_min;
        range_f(range_f == 0) = 1;
        costs_norm = (costs - f_min) ./ range_f;
        
        % Calculate Spacing (Normalized)
        if size(costs_norm, 1) > 1
            spacing = compute_spacing(costs_norm);
            performance_history.spacing(iteration) = spacing;
        else
            performance_history.spacing(iteration) = 0;
        end
        
        % Calculate Convergence (Normalized)
        convergence = mean(sqrt(sum(costs_norm.^2, 2)));
        performance_history.convergence(iteration) = convergence;
        
        % Calculate HV (Normalized)
        ref_norm = [1.1, 1.1];
        hv = hv2d_min(costs_norm, ref_norm);
        performance_history.hv(iteration) = hv;
        
    else
        performance_history.spacing(iteration) = 0;
        performance_history.convergence(iteration) = inf;
        performance_history.hv(iteration) = 0;
    end
end

function spacing = compute_spacing(costs)
    % Calculate SPACING metric - Measures distribution uniformity of Pareto front solutions
    % Smaller SPACING value means more uniform distribution
    if size(costs, 1) < 2
        spacing = 0;
        return;
    end
    
    n = size(costs, 1);
    distances = zeros(n, 1);
    
    % Calculate distance from each solution to its nearest neighbor
    for i = 1:n
        min_dist = inf;
        for j = 1:n
            if i ~= j
                % Calculate Euclidean distance
                dist = norm(costs(i,:) - costs(j,:));
                if dist < min_dist
                    min_dist = dist;
                end
            end
        end
        distances(i) = min_dist;
    end
    
    % SPACING metric: standard deviation of distances
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
function v = igd_min(costs, refset)
if isempty(costs) || isempty(refset)
    v = inf;
    return
end
v = 0;
for i = 1:size(refset,1)
    d = sqrt(sum((costs - refset(i,:)).^2,2));
    v = v + min(d);
end
v = v / size(refset,1);
end
