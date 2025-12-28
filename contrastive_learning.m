function cl_data = contrastive_learning(pop, F, temperature, alpha)
% Contrastive learning module - optimize population evolution for NSGA2
% Input:
%   pop - Current population
%   F - Non-dominated sorting results
%   temperature - Temperature parameter, controls sensitivity of similarity calculation
%   alpha - Learning rate parameter
% Output:
%   cl_data - Structure containing feature representation, similarity matrix and contrastive loss

    if nargin < 3
        temperature = 0.1;  % Default temperature parameter
    end
    if nargin < 4
        alpha = 0.5;  % Default learning rate
    end
    
    npop = length(pop);
    nvar = length(pop(1).position);
    
    % Initialize contrastive learning data structure
    cl_data = struct();
    cl_data.features = zeros(npop, nvar + 2);  % Position + Objective function values
    cl_data.similarity_matrix = zeros(npop, npop);
    cl_data.positive_pairs = [];
    cl_data.negative_pairs = [];
    cl_data.contrastive_loss = 0;
    
    %% 1. Feature extraction and representation learning
    for i = 1:npop
        % Combine position vector and objective function values as features
        cl_data.features(i, :) = [pop(i).position, pop(i).cost'];
    end
    
    % Feature normalization
    cl_data.features = normalize_features(cl_data.features);
    
    %% 2. Construct positive and negative sample pairs
    [cl_data.positive_pairs, cl_data.negative_pairs] = construct_pairs(pop, F);
    
    %% 3. Calculate similarity matrix
    cl_data.similarity_matrix = compute_similarity_matrix(cl_data.features, temperature);
    
    %% 4. Calculate contrastive loss
    cl_data.contrastive_loss = compute_contrastive_loss(cl_data.similarity_matrix, ...
                                                       cl_data.positive_pairs, ...
                                                       cl_data.negative_pairs, ...
                                                       temperature);
    
    %% 5. Calculate gradient information to guide evolutionary operations
    cl_data.gradients = compute_gradients(cl_data.features, cl_data.positive_pairs, ...
                                         cl_data.negative_pairs, alpha);
    
    %% 6. Calculate individual quality score
    cl_data.quality_scores = compute_quality_scores(pop, cl_data.similarity_matrix, F);
end

%% Helper functions
function normalized_features = normalize_features(features)
    % Feature normalization function
    normalized_features = zeros(size(features));
    for i = 1:size(features, 2)
        col = features(:, i);
        if std(col) > 1e-8  % Avoid division by zero
            normalized_features(:, i) = (col - mean(col)) / std(col);
        else
            normalized_features(:, i) = col;
        end
    end
end

function [positive_pairs, negative_pairs] = construct_pairs(pop, F)
    % Construct positive and negative sample pairs
    % Positive pairs: Individuals in the same non-dominated level
    % Negative pairs: Individuals in different non-dominated levels
    
    positive_pairs = [];
    negative_pairs = [];
    
    npop = length(pop);
    
    % Construct positive pairs (individuals within the same level)
    for level = 1:length(F)
        level_indices = F{level};
        n_level = length(level_indices);
        
        % Construct positive pairs within the same level
        for i = 1:n_level-1
            for j = i+1:n_level
                positive_pairs = [positive_pairs; level_indices(i), level_indices(j)];
            end
        end
    end
    
    % Construct negative pairs (individuals between different levels)
    for level1 = 1:length(F)-1
        for level2 = level1+1:length(F)
            indices1 = F{level1};
            indices2 = F{level2};
            
            % Randomly select some negative pairs to control computational complexity
            n_neg_samples = min(length(indices1) * length(indices2), 50);
            
            for k = 1:n_neg_samples
                i = indices1(randi(length(indices1)));
                j = indices2(randi(length(indices2)));
                negative_pairs = [negative_pairs; i, j];
            end
        end
    end
end

function similarity_matrix = compute_similarity_matrix(features, temperature)
    % Calculate similarity matrix
    npop = size(features, 1);
    similarity_matrix = zeros(npop, npop);
    
    for i = 1:npop
        for j = 1:npop
            if i ~= j
                % Use cosine similarity
                cosine_sim = dot(features(i,:), features(j,:)) / ...
                           (norm(features(i,:)) * norm(features(j,:)) + 1e-8);
                
                % Apply temperature parameter
                similarity_matrix(i, j) = exp(cosine_sim / temperature);
            end
        end
    end
end

function loss = compute_contrastive_loss(similarity_matrix, positive_pairs, negative_pairs, temperature)
    % Calculate contrastive loss function
    loss = 0;
    n_pos = size(positive_pairs, 1);
    n_neg = size(negative_pairs, 1);
    
    if n_pos == 0 || n_neg == 0
        return;
    end
    
    % Loss of positive pairs
    pos_loss = 0;
    for k = 1:n_pos
        i = positive_pairs(k, 1);
        j = positive_pairs(k, 2);
        
        % Calculate denominator: sum of similarities with all other samples
        denominator = sum(similarity_matrix(i, :)) - similarity_matrix(i, i);
        
        if denominator > 1e-8
            pos_loss = pos_loss - log(similarity_matrix(i, j) / denominator);
        end
    end
    
    loss = pos_loss / n_pos;
end

function gradients = compute_gradients(features, positive_pairs, negative_pairs, alpha)
    % Calculate gradient information to guide evolutionary operations
    npop = size(features, 1);
    nvar = size(features, 2);
    gradients = zeros(npop, nvar);
    
    % Calculate gradient direction based on positive and negative sample pairs
    for k = 1:size(positive_pairs, 1)
        i = positive_pairs(k, 1);
        j = positive_pairs(k, 2);
        
        % Positive pairs should be more similar, gradient points to each other
        direction = features(j, :) - features(i, :);
        gradients(i, :) = gradients(i, :) + alpha * direction;
        gradients(j, :) = gradients(j, :) - alpha * direction;
    end
    
    for k = 1:size(negative_pairs, 1)
        i = negative_pairs(k, 1);
        j = negative_pairs(k, 2);
        
        % Negative pairs should be more dissimilar, gradient moves away from each other
        direction = features(j, :) - features(i, :);
        gradients(i, :) = gradients(i, :) - alpha * direction;
        gradients(j, :) = gradients(j, :) + alpha * direction;
    end
    
    % Gradient normalization
    for i = 1:npop
        if norm(gradients(i, :)) > 1e-8
            gradients(i, :) = gradients(i, :) / norm(gradients(i, :));
        end
    end
end

function quality_scores = compute_quality_scores(pop, similarity_matrix, F)
    % Calculate individual quality score, combining non-dominated sorting and similarity information
    npop = length(pop);
    quality_scores = zeros(npop, 1);
    
    for i = 1:npop
        % Base score: non-dominated level (lower level means higher score)
        base_score = 1.0 / (pop(i).rank + 1);
        
        % Diversity score: average similarity with other individuals (lower is better)
        avg_similarity = mean(similarity_matrix(i, :));
        diversity_score = 1.0 / (1.0 + avg_similarity);
        
        % Crowding distance score
        crowding_score = pop(i).crowdingdistance / (1.0 + pop(i).crowdingdistance);
        
        % Comprehensive quality score
        quality_scores(i) = 0.5 * base_score + 0.3 * diversity_score + 0.2 * crowding_score;
    end
end