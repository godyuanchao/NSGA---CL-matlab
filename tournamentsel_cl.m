function p = tournamentsel_cl(pop, cl_data, tournament_size)
% Contrastive Learning Enhanced Tournament Selection Operator
% Input:
%   pop - Current population
%   cl_data - Contrastive learning data structure
%   tournament_size - Tournament size (default 2)
% Output:
%   p - Selected individual
    
    if nargin < 3
        tournament_size = 2;
    end
    
    n = numel(pop);
    
    % If no contrastive learning data, use standard selection
    if nargin < 2 || isempty(cl_data)
        p = standard_tournament_selection(pop, tournament_size);
        return;
    end
    
    if evalin('base','exist(''CL_VARIANT'',''var'')')
        vmode = evalin('base','CL_VARIANT');
        if strcmpi(vmode,'mutation_only') || strcmpi(vmode,'crossover_only')
            p = standard_tournament_selection(pop, tournament_size);
            return
        end
    end
    
    % Use contrastive learning enhanced selection strategy
    p = contrastive_tournament_selection(pop, cl_data, tournament_size);
end

function p = contrastive_tournament_selection(pop, cl_data, tournament_size)
    % Tournament Selection based on Contrastive Learning
    n = numel(pop);
    
    % Randomly select tournament candidates
    candidates = randperm(n, tournament_size);
    
    % Calculate comprehensive fitness score for each candidate
    fitness_scores = zeros(tournament_size, 1);
    
    for i = 1:tournament_size
        idx = candidates(i);
        fitness_scores(i) = compute_comprehensive_fitness(pop(idx), cl_data, idx);
    end
    
    % Select the individual with the highest fitness
    [~, best_idx] = max(fitness_scores);
    p = pop(candidates(best_idx));
end

function fitness = compute_comprehensive_fitness(individual, cl_data, idx)
    % Calculate comprehensive fitness score, combining traditional NSGA2 metrics and contrastive learning info
    
    % 1. Traditional NSGA2 fitness (based on rank and crowding distance)
    % Amplify Rank weight to ensure hierarchy priority (Rank 1 -> 5.0, Rank 2 -> 3.33)
    rank_fitness = 10.0 / (individual.rank + 1);  
    crowding_fitness = individual.crowdingdistance / (1.0 + individual.crowdingdistance);
    
    % 2. Contrastive Learning Quality Score
    quality_score = cl_data.quality_scores(idx);
    
    % 3. Diversity Score (based on similarity matrix)
    diversity_score = compute_diversity_score(cl_data.similarity_matrix, idx);
    
    % 4. Gradient Information Score (larger gradient means greater improvement potential)
    gradient_score = compute_gradient_score(cl_data.gradients, idx);
    
    % Comprehensive fitness score (weighted combination)
    w1 = 1.0;  % Traditional NSGA2 weight (dominant)
    w2 = 0.0;  % Disable quality weight
    w3 = 0.4;  % Diversity weight (auxiliary Crowding)
    w4 = 0.0;  % Disable gradient weight
    
    % In traditional fitness, Rank is dominant, Crowding is auxiliary
    traditional_fitness = rank_fitness + 0.5 * crowding_fitness; 
    
    fitness = w1 * traditional_fitness + w2 * quality_score + ...
              w3 * diversity_score + w4 * gradient_score;
end

function diversity_score = compute_diversity_score(similarity_matrix, idx)
    % Calculate diversity score of the individual
    % The lower the average similarity with other individuals, the higher the diversity score
    
    avg_similarity = mean(similarity_matrix(idx, :));
    diversity_score = 1.0 / (1.0 + avg_similarity);
end

function gradient_score = compute_gradient_score(gradients, idx)
    % Calculate gradient score
    % The larger the gradient magnitude, the greater the improvement potential
    
    gradient_magnitude = norm(gradients(idx, :));
    gradient_score = gradient_magnitude / (1.0 + gradient_magnitude);
end

function p = standard_tournament_selection(pop, tournament_size)
    % Standard Tournament Selection (Original Version)
    n = numel(pop);
    candidates = randperm(n, tournament_size);
    
    best_candidate = candidates(1);
    best_individual = pop(best_candidate);
    
    for i = 2:tournament_size
        current_candidate = candidates(i);
        current_individual = pop(current_candidate);
        
        % Compare Rank
        if current_individual.rank < best_individual.rank
            best_candidate = current_candidate;
            best_individual = current_individual;
        elseif current_individual.rank == best_individual.rank
            % Compare Crowding Distance if ranks are equal
            if current_individual.crowdingdistance > best_individual.crowdingdistance
                best_candidate = current_candidate;
                best_individual = current_individual;
            end
        end
    end
    
    p = best_individual;
end

% Advanced Selection Strategy Function
function p = adaptive_tournament_selection(pop, cl_data, generation, max_generation)
    % Adaptive Tournament Selection: Adjust selection pressure based on evolutionary stage
    
    % Calculate evolutionary progress
    progress = generation / max_generation;
    
    % Early stage: smaller tournament size, maintain diversity
    % Late stage: larger tournament size, increase selection pressure
    if progress < 0.3
        tournament_size = 2;
        diversity_weight = 0.4;
    elseif progress < 0.7
        tournament_size = 3;
        diversity_weight = 0.3;
    else
        tournament_size = 4;
        diversity_weight = 0.2;
    end
    
    n = numel(pop);
    candidates = randperm(n, tournament_size);
    
    % Calculate adaptive fitness score for each candidate
    fitness_scores = zeros(tournament_size, 1);
    
    for i = 1:tournament_size
        idx = candidates(i);
        fitness_scores(i) = compute_adaptive_fitness(pop(idx), cl_data, idx, diversity_weight);
    end
    
    % Select the individual with the highest fitness
    [~, best_idx] = max(fitness_scores);
    p = pop(candidates(best_idx));
end

function fitness = compute_adaptive_fitness(individual, cl_data, idx, diversity_weight)
    % Calculate adaptive fitness score
    
    % Basic fitness
    rank_fitness = 1.0 / (individual.rank + 1);
    crowding_fitness = individual.crowdingdistance / (1.0 + individual.crowdingdistance);
    quality_score = cl_data.quality_scores(idx);
    diversity_score = compute_diversity_score(cl_data.similarity_matrix, idx);
    
    % Adaptive weights
    quality_weight = 1.0 - diversity_weight;
    
    fitness = 0.5 * (0.7 * rank_fitness + 0.3 * crowding_fitness) + ...
              quality_weight * quality_score + diversity_weight * diversity_score;
end

% Multi-objective Selection Strategy
function p = multi_objective_selection(pop, cl_data)
    % Multi-objective Selection: Consider both convergence and diversity
    
    n = numel(pop);
    
    % Calculate multi-objective score for each individual
    convergence_scores = zeros(n, 1);
    diversity_scores = zeros(n, 1);
    
    for i = 1:n
        % Convergence score
        convergence_scores(i) = 1.0 / (pop(i).rank + 1) + cl_data.quality_scores(i);
        
        % Diversity score
        diversity_scores(i) = compute_diversity_score(cl_data.similarity_matrix, i);
    end
    
    % Normalize scores
    convergence_scores = convergence_scores / max(convergence_scores);
    diversity_scores = diversity_scores / max(diversity_scores);
    
    % Calculate Pareto front
    pareto_front = find_pareto_front(convergence_scores, diversity_scores);
    
    % Randomly select from Pareto front
    if ~isempty(pareto_front)
        selected_idx = pareto_front(randi(length(pareto_front)));
    else
        % If no Pareto front, use standard selection
        selected_idx = randi(n);
    end
    
    p = pop(selected_idx);
end

function pareto_front = find_pareto_front(obj1, obj2)
    % Find Pareto front in 2D objective space
    n = length(obj1);
    pareto_front = [];
    
    for i = 1:n
        is_dominated = false;
        for j = 1:n
            if i ~= j
                % Check if i is dominated by j
                if (obj1(j) >= obj1(i) && obj2(j) >= obj2(i)) && ...
                   (obj1(j) > obj1(i) || obj2(j) > obj2(i))
                    is_dominated = true;
                    break;
                end
            end
        end
        
        if ~is_dominated
            pareto_front = [pareto_front, i];
        end
    end
end
