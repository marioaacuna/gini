function indexes = calculate_indexes(predictors, ori_table)
    
    weights = predictors.weights;
    best_predictors = predictors.names;
    n_predictors = length(best_predictors);
    %Index = w1 * MaxV_1 + w2 * MaxOrder + w3 * SAG
    
    pred = [];
    for ip = 1:n_predictors
        pred(:,ip) = weights(ip)* table2array(ori_table(:,best_predictors(:,ip)));
    end
    indexes = sum(pred, 2);


