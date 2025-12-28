function pop = Sortpop(pop)
    
    [~, CDSO] = sort([pop.crowdingdistance], 'descend'); % Sort descending
    pop = pop(CDSO);
    
    [~, RSO] = sort([pop.rank]); % Sort ascending
    pop = pop(RSO);


end
