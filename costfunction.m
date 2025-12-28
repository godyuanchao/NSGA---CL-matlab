function z = costfunction(x)
    % Example Objective Function (e.g., Response Surface Model)
    % x: Decision variables
    % z: Objective values (to be minimized)

    % Objective 1 (Generic polynomial example)
    f1 = -42.480 + 9.225*x(1) +1.556*x(2) -0.096*x(3) ...
        + 0.321*x(1)*x(2) + 0.003*x(1)*x(3) + 0.003*x(2)*x(3) ...
        - 5.17*x(1)^2-0.023*x(2)^2 ;

    % Objective 2 (Generic polynomial example)
    f2 = -99220.394 + 21281.174*x(1) - 516.795*x(2) + 4355.944*x(3) ...
        -1204.733*x(1)*x(2) -969.171*x(1)*x(3) - 29.758*x(2)*x(3) ...
        + 44284.439*x(1)^2 + 53.644*x(2)^2 - 12.315*x(3)^2;
        
    z = [-f1; f2]; % Note: f1 is negated for minimization if the original problem required maximizing f1
end