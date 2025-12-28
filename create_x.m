function x = create_x(var)
    % Initialize individual with discrete/stepped variables
    % var: [min; step; max; num_steps]
    
    n = size(var, 2);
    x = zeros(1, n);
    for i = 1 : n
        % Randomly select a step index and calculate value
        x(i) = var(1, i) + randi([0, var(4, i)], 1) * var(2, i);
    end
end