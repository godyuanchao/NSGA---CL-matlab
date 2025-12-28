function plotcosts(pop)

    costs = [pop.cost];
    
    plot(costs(1, :), costs(2, :), 'r*', 'MarkerSize', 10);
    xlabel('Objective 1');
    ylabel('Objective 2');
    title('Pareto Front');
    grid on;

end
