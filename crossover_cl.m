function [y1, y2] = crossover_cl(x1, x2, cl_data, p1_idx, p2_idx, var)
% Enhanced crossover operator based on contrastive learning
% Input:
%   x1, x2 - Position vectors of parent individuals
%   cl_data - Contrastive learning data structure
%   p1_idx, p2_idx - Indices of parent individuals in the population
%   var - Variable constraint info [varmin; step; varmax; len]
% Output:
%   y1, y2 - Position vectors of offspring individuals

    n = numel(x1);
    
    if evalin('base','exist(''CL_VARIANT'',''var'')')
        vmode = evalin('base','CL_VARIANT');
        if strcmpi(vmode,'selection_only') || strcmpi(vmode,'mutation_only')
            [y1, y2] = standard_crossover(x1, x2);
            if nargin >= 6
                y1 = enforce_bounds(y1, var);
                y2 = enforce_bounds(y2, var);
            end
            return
        end
    end
    
    % Get gradient information from contrastive learning
    if nargin >= 3 && ~isempty(cl_data) && nargin >= 5
        grad1 = cl_data.gradients(p1_idx, 1:n);  % Take only the position part of the gradient
        grad2 = cl_data.gradients(p2_idx, 1:n);
        
        % Calculate similarity between parent individuals
        similarity = cl_data.similarity_matrix(p1_idx, p2_idx);
        
        % Adjust crossover strategy based on similarity
        if similarity > 0.7  % High similarity, use exploratory crossover
            [y1, y2] = exploratory_crossover(x1, x2, grad1, grad2, var);
        elseif similarity < 0.3  % Low similarity, use conservative crossover
            [y1, y2] = conservative_crossover(x1, x2, var);
        else  % Medium similarity, use gradient-guided crossover
            [y1, y2] = gradient_guided_crossover(x1, x2, grad1, grad2, var);
        end
    else
        % Fallback to original crossover method
        [y1, y2] = standard_crossover(x1, x2);
    end
    
    % Ensure results are within constraints
    if nargin >= 6
        y1 = enforce_bounds(y1, var);
        y2 = enforce_bounds(y2, var);
    end
end

function [y1, y2] = exploratory_crossover(x1, x2, grad1, grad2, var)
    % Exploratory crossover: Used for similar individuals to increase diversity
    n = numel(x1);
    
    % Use gradient information to guide crossover point selection
    grad_magnitude = abs(grad1) + abs(grad2);
    [~, sorted_idx] = sort(grad_magnitude, 'descend');
    
    % Select dimensions with larger gradients for crossover
    crossover_prob = 0.7;  % Higher crossover probability
    crossover_mask = rand(1, n) < crossover_prob;
    
    % Prioritize crossover in dimensions with large gradients
    priority_dims = sorted_idx(1:ceil(n/2));
    crossover_mask(priority_dims) = true;
    
    y1 = x1;
    y2 = x2;
    
    % Execute crossover
    y1(crossover_mask) = x2(crossover_mask);
    y2(crossover_mask) = x1(crossover_mask);
    
    % Add random perturbation to increase diversity
    perturbation_strength = 0.1;
    if nargin >= 5
        varmin = var(1, :);
        varmax = var(3, :);
        range_size = varmax - varmin;
        
        perturbation1 = perturbation_strength * range_size .* (2*rand(1,n) - 1);
        perturbation2 = perturbation_strength * range_size .* (2*rand(1,n) - 1);
        
        y1 = y1 + perturbation1;
        y2 = y2 + perturbation2;
    end
end

function [y1, y2] = conservative_crossover(x1, x2, var)
    % Conservative crossover: Used for individuals with large differences to maintain stability
    n = numel(x1);
    
    % Use lower crossover probability
    crossover_prob = 0.3;
    crossover_mask = rand(1, n) < crossover_prob;
    
    y1 = x1;
    y2 = x2;
    
    % Execute crossover
    y1(crossover_mask) = x2(crossover_mask);
    y2(crossover_mask) = x1(crossover_mask);
    
    % Use weighted average for smoothing
    alpha = 0.3;  % Smoothing coefficient
    y1 = alpha * y1 + (1 - alpha) * x1;
    y2 = alpha * y2 + (1 - alpha) * x2;
end

function [y1, y2] = gradient_guided_crossover(x1, x2, grad1, grad2, var)
    % Gradient-guided crossover: Use contrastive learning gradient information to guide crossover
    n = numel(x1);
    
    % Calculate gradient direction consistency
    grad_consistency = grad1 .* grad2;
    
    % Perform crossover in dimensions where gradient directions are consistent
    consistent_dims = grad_consistency > 0;
    inconsistent_dims = grad_consistency <= 0;
    
    y1 = x1;
    y2 = x2;
    
    % Perform standard crossover in dimensions with consistent gradients
    if sum(consistent_dims) > 0
        crossover_mask = rand(1, sum(consistent_dims)) < 0.6;
        temp_dims = find(consistent_dims);
        selected_dims = temp_dims(crossover_mask);
        
        y1(selected_dims) = x2(selected_dims);
        y2(selected_dims) = x1(selected_dims);
    end
    
    % Use gradient-guided interpolation in dimensions with inconsistent gradients
    if sum(inconsistent_dims) > 0
        inconsistent_idx = find(inconsistent_dims);
        for i = inconsistent_idx
            % Weighted interpolation based on gradient magnitude
            w1 = abs(grad1(i)) / (abs(grad1(i)) + abs(grad2(i)) + 1e-8);
            w2 = 1 - w1;
            
            y1(i) = w1 * x1(i) + w2 * x2(i);
            y2(i) = w2 * x1(i) + w1 * x2(i);
            
            % Add small random perturbation to maintain diversity
             if nargin >= 5
                 range_i = var(3, i) - var(1, i);
                 perturb = 0.05 * range_i * (rand - 0.5);
                 y1(i) = y1(i) + perturb;
                 y2(i) = y2(i) - perturb;
             end
        end
    end
    
    % Add gradient-based fine-tuning
    learning_rate = 0.01; % Reduce learning rate to prevent over-convergence to local regions (was 0.05)
    if nargin >= 5
        varmin = var(1, :);
        varmax = var(3, :);
        range_size = varmax - varmin;
        
        % Fine-tune along gradient direction
        adjustment1 = learning_rate * range_size .* grad1;
        adjustment2 = learning_rate * range_size .* grad2;
        
        y1 = y1 + adjustment1;
        y2 = y2 + adjustment2;
    end
end

function [y1, y2] = standard_crossover(x1, x2)
    % Standard crossover operator (original version)
    n = numel(x1);
    r = randi([1, n - 1], 1);
    index = randperm(r, r);

    y1 = x1;
    y1(index) = x2(index);

    y2 = x2;
    y2(index) = x1(index);
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
