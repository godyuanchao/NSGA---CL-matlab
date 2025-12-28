if ~exist('is_subroutine', 'var')
    clc;
    clear;
    close all;
    rng(40);
end
%% Example Problem Configuration (Design Variables)
nvar = 3;           % Number of design variables
nobj = 2;           % Number of objective functions
npop = 50;
maxit = 100;
pc = 0.8;
nc = round(pc * npop / 2) * 2;
mu = 0.05;
 varmin = [1.5,40,30];% Lower bound
 varmax = [2.5,50,75];% Upper bound
 step = [0.01, 1, 1];% Step size
len = (varmax - varmin) ./ step;
var = [varmin;step;varmax;round(len, 0)];
%% Define result storage template
empty.position = [];
empty.cost = [];
empty.rank = [];
empty.domination = [];
empty.dominated = 0;
empty.crowdingdistance = [];
pop = repmat(empty, npop, 1);

% Performance record
performance_history = struct();
performance_history.convergence_generation = [];
performance_history.spacing = [];
performance_history.convergence = [];
performance_history.hv = [];
performance_history.hv_final = [];

% Initialize population
for i = 1 : npop
    pop(i).position = create_x(var);
    pop(i).cost = costfunction(pop(i).position);
end

%% Non-dominated sorting
[pop,F] = nondominatedsort(pop);

%% Crowding distance calculation
pop = calcrowdingdistance(pop,F);

%% Main program
for it = 1 : maxit
    
    popc = repmat(empty, nc/2,2);
    
    for j = 1 : nc / 2
       p1 = tournamentsel(pop);
       p2 = tournamentsel(pop);
       [popc(j, 1).position, popc(j, 2).position] = crossover(p1.position, p2.position);
    end
    
    popc = popc(:);
    
    for k = 1 : nc
        popc(k).position = mutate(popc(k).position, mu, var);
        popc(k).cost = costfunction(popc(k).position);
    end
   
    newpop = [pop; popc];
    
    [pop,F] = nondominatedsort(newpop);

    pop = calcrowdingdistance(pop,F);
    
    % Sort
    pop = Sortpop(pop);
    
    % Eliminate
    pop = pop(1: npop);

    [pop,F] = nondominatedsort(pop);

    pop = calcrowdingdistance(pop,F);
    
    pop = Sortpop(pop);
    
    % Update rank 1
    F1 = pop(F{1});
    
    %% Record performance metrics
    performance_history = update_performance_metrics(performance_history, F1, pop, it);
    
    % Display iteration information
    % Plotting - Only keep the final figure5 output
end


%% Result analysis and output
F1_final = pop(F{1});

%% Performance metric output

% Calculate final performance metrics
final_convergence_generation = performance_history.convergence_generation(end);
final_spacing = performance_history.spacing(end);
final_convergence = performance_history.convergence(end);

% Output performance metrics

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

% Output statistical information
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
    
    % 6. Calculate normalized IGD
    % Note: Cannot calculate true IGD in single run (requires known Pareto front or global reference set)
    % Set to 0 here, true IGD needs to be calculated in compare_algorithms.m using global reference set
    final_igd = 0;
    
    % 7. Calculate normalized SPACING
    final_spacing = compute_spacing_traditional(costs_norm);
    
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
    
    assignin('base','baseline_final_spacing', final_spacing);
    assignin('base','baseline_final_convergence', final_convergence);
    assignin('base','baseline_final_hv', final_hv);
    
    if exist('costsF', 'var')
        assignin('base', 'costsF', costsF);
    end



% Optimal solution info saved to result file

% Save results
if evalin('base','exist(''results_filename'',''var'')')
    rf = evalin('base','results_filename');
    save(rf, 'pop', 'F', 'performance_history');
else
    save('nsga2_traditional_results.mat', 'pop', 'F', 'performance_history');
end

% Save Spacing, Convergence, HV of each generation to CSV
metrics_table = table((1:length(performance_history.spacing))', ...
                      performance_history.spacing(:), ...
                      performance_history.convergence(:), ...
                      performance_history.hv(:), ...
                      'VariableNames', {'Generation', 'Spacing', 'Convergence', 'HV'});
writetable(metrics_table, 'nsga2_metrics.csv');


% Output figure5 numerical data (sorted by Objective_1 ascending)
costs = [F1_final.cost];

% Extract data and transform Objective_1 values
obj1_values = -costs(1, :);
obj2_values = costs(2, :);

% Sort by Objective_1 values in ascending order
[sorted_obj1, sorted_indices] = sort(obj1_values);
sorted_obj2 = obj2_values(sorted_indices);

% Output sorted data

% Save Pareto front data to CSV (appending to nsga2_metrics.csv is not appropriate due to different dimensions, suggest creating new file or new columns but lengths differ)
% For convenience, we create a separate Pareto data file
pareto_table = table(sorted_obj1(:), sorted_obj2(:), ...
                     'VariableNames', {'Objective_1', 'Objective_2'});
writetable(pareto_table, 'nsga2_pareto_front.csv');
if ~exist('is_subroutine', 'var') || ~is_subroutine
    fprintf('Iteration metrics saved to nsga2_metrics.csv\n');
    fprintf('Pareto front data saved to nsga2_pareto_front.csv\n');
end

%% Helper functions
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
function performance_history = update_performance_metrics(performance_history, F1, pop, iteration)
    % Record convergence generation
    performance_history.convergence_generation(iteration) = iteration;
    
    % Calculate SPACING metric
    % Update performance metrics
    if ~isempty(F1)
        costs = [F1.cost]';
        
        % Normalize costs first
        f_min = min(costs, [], 1);
        f_max = max(costs, [], 1);
        range_f = f_max - f_min;
        range_f(range_f == 0) = 1;
        costs_norm = (costs - f_min) ./ range_f;
        
        % Calculate Spacing (Normalized)
        if size(costs_norm, 1) > 1
            spacing = compute_spacing_traditional(costs_norm);
            performance_history.spacing(iteration) = spacing;
        else
            performance_history.spacing(iteration) = 0;
        end
        
        % Calculate Convergence (Normalized distance to origin)
        convergence_metric = mean(sqrt(sum(costs_norm.^2, 2)));
        performance_history.convergence(iteration) = convergence_metric;
        
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

function spacing = compute_spacing_traditional(costs)
    % Calculate SPACING metric - Measures distribution uniformity of Pareto front solutions
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
                dist = norm(costs(i,:) - costs(j,:));
                if dist < min_dist
                    min_dist = dist;
                end
            end
        end
        distances(i) = min_dist;
    end
    
    % SPACING metric is the standard deviation of distances
    mean_dist = mean(distances);
    spacing = sqrt(sum((distances - mean_dist).^2) / n);
end
