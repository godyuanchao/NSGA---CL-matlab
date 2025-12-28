function y = mutate_cl(x, mu, var, cl_data, individual_idx)
% Enhanced mutation operator based on contrastive learning
% Input:
%   x - Position vector of the individual
%   mu - Mutation probability
%   var - Variable constraint info [varmin; step; varmax; len]
%   cl_data - Contrastive learning data structure
%   individual_idx - Index of the individual in the population
% Output:
%   y - Position vector of the mutated individual

    n = numel(x);
    
    if evalin('base','exist(''CL_VARIANT'',''var'')')
        vmode = evalin('base','CL_VARIANT');
        if strcmpi(vmode,'selection_only') || strcmpi(vmode,'crossover_only')
            y = standard_mutate(x, mu, var);
            return
        end
    end
    
    % If no contrastive learning data, use standard mutation
    if nargin < 4 || isempty(cl_data) || nargin < 5
        y = standard_mutate(x, mu, var);
        return;
    end
    
    % Get contrastive learning information
    gradient = cl_data.gradients(individual_idx, 1:n);  % Take only the position part of the gradient
    quality_score = cl_data.quality_scores(individual_idx);
    
    % Adjust mutation strategy based on individual quality
    if quality_score > 0.7  % High quality individual, use fine-tuning mutation
        y = fine_tuning_mutate(x, mu, var, gradient);
    elseif quality_score < 0.3  % Low quality individual, use exploratory mutation
        y = exploratory_mutate(x, mu, var, gradient);
    else  % Medium quality individual, use gradient-guided mutation
        y = gradient_guided_mutate(x, mu, var, gradient);
    end
    
    % Ensure results are within constraints
    y = enforce_bounds(y, var);
end

function y = fine_tuning_mutate(x, mu, var, gradient)
    % Fine-tuning mutation: Used for high-quality individuals, perform small precise adjustments
    n = numel(x);
    varmin = var(1, :);
    varmax = var(3, :);
    range_size = varmax - varmin;
    
    % Decrease mutation probability, increase precision
    adjusted_mu = mu * 0.5;  % Decrease mutation probability
    r = rand(1, n);
    mutate_mask = r <= adjusted_mu;
    
    y = x;
    
    if sum(mutate_mask) > 0
        % Gradient-based fine-tuning
        learning_rate = 0.02;  % Smaller learning rate
        
        % Small adjustment along gradient direction
        gradient_adjustment = learning_rate * range_size .* gradient;
        
        % Add small random perturbation
        random_perturbation = 0.01 * range_size .* (2*rand(1,n) - 1);
        
        % Combined adjustment
        total_adjustment = 0.7 * gradient_adjustment + 0.3 * random_perturbation;
        
        y(mutate_mask) = y(mutate_mask) + total_adjustment(mutate_mask);
    end
end

function y = exploratory_mutate(x, mu, var, gradient)
    % Exploratory mutation: Used for low-quality individuals, perform large-scale exploration
    n = numel(x);
    varmin = var(1, :);
    varmax = var(3, :);
    range_size = varmax - varmin;
    
    % Increase mutation probability and strength
    adjusted_mu = min(mu * 2.0, 0.8);  % Increase mutation probability but not exceeding 0.8
    r = rand(1, n);
    mutate_mask = r <= adjusted_mu;
    
    y = x;
    
    if sum(mutate_mask) > 0
        % Large-scale random mutation
        exploration_strength = 0.3;  % Larger exploration strength
        
        % Anti-gradient direction exploration (escape current local region)
        anti_gradient_adjustment = -exploration_strength * range_size .* gradient;
        
        % Large random perturbation
        random_exploration = exploration_strength * range_size .* (2*rand(1,n) - 1);
        
        % Combined exploration
        total_adjustment = 0.4 * anti_gradient_adjustment + 0.6 * random_exploration;
        
        y(mutate_mask) = y(mutate_mask) + total_adjustment(mutate_mask);
        
        % Completely reinitialize some dimensions
        reinit_prob = 0.2;
        reinit_mask = rand(1, n) < reinit_prob;
        reinit_mask = reinit_mask & mutate_mask;
        
        if sum(reinit_mask) > 0
            y(reinit_mask) = varmin(reinit_mask) + ...
                           (varmax(reinit_mask) - varmin(reinit_mask)) .* rand(1, sum(reinit_mask));
        end
    end
end

function y = gradient_guided_mutate(x, mu, var, gradient)
    % Gradient-guided mutation: Use contrastive learning gradient information to guide mutation direction
    n = numel(x);
    varmin = var(1, :);
    varmax = var(3, :);
    range_size = varmax - varmin;
    
    r = rand(1, n);
    mutate_mask = r <= mu;
    
    y = x;
    
    if sum(mutate_mask) > 0
        % Calculate gradient magnitude
        gradient_magnitude = abs(gradient);
        max_grad = max(gradient_magnitude);
        
        if max_grad > 1e-8
            % Normalize gradient magnitude
            normalized_grad_mag = gradient_magnitude / max_grad;
            
            % Adjust mutation strength based on gradient magnitude
            adaptive_strength = 0.05 + 0.15 * normalized_grad_mag;
            
            % Adjustment along gradient direction
            gradient_adjustment = adaptive_strength .* range_size .* gradient;
            
            % Random perturbation perpendicular to gradient direction (increase diversity)
            orthogonal_perturbation = 0.05 * range_size .* (2*rand(1,n) - 1);
            
            % Combined adjustment
            total_adjustment = 0.8 * gradient_adjustment + 0.2 * orthogonal_perturbation;
            
            y(mutate_mask) = y(mutate_mask) + total_adjustment(mutate_mask);
        else
            % If gradient is very small, use standard random mutation
            new_values = create_x(var);
            y(mutate_mask) = new_values(mutate_mask);
        end
    end
end

function y = standard_mutate(x, mu, var)
    % Standard mutation operator (original version)
    n = numel(x);
    r = rand(1, n);
    index = find(r <= mu);
    
    if ~isempty(index)
        new = create_x(var);
        x(index) = new(index);
    end
    
    y = x;
end

function y = enforce_bounds(x, var)
    % Ensure variables are within constraints
    if nargin < 2
        y = x;
        return;
    end
    
    varmin = var(1, :);
    step = var(2, :);
    varmax = var(3, :);
    
    y = x;
    
    % Boundary constraints
    y = max(y, varmin);
    y = min(y, varmax);
    
    % Step size constraints (discretization)
    for i = 1:length(y)
        if step(i) > 0
            y(i) = varmin(i) + round((y(i) - varmin(i)) / step(i)) * step(i);
        end
    end
end

% Helper function: Adaptive mutation strength calculation
function strength = compute_adaptive_strength(gradient, base_strength)
    % Calculate adaptive mutation strength based on gradient information
    gradient_norm = norm(gradient);
    
    if gradient_norm > 1e-8
        % Larger gradient means larger mutation strength (needs larger adjustment)
        strength = base_strength * (1 + gradient_norm);
    else
        strength = base_strength;
    end
    
    % Limit mutation strength range
    strength = min(strength, 0.5);  % Max not exceeding 50% range
    strength = max(strength, 0.01); % Min keeping 1% range
end
