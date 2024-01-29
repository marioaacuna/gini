function [pred_names, best_predictors, best_pred_vals, idx]=load_best_predictors()
global GC

%% Load Tree estimatiors
pred_filename = fullfile(GC.raw_data_folder, 'out', 'TREE_predictors.mat');
preds = load_variable(pred_filename, 'TREE_predictors');
S = preds.S;
norm_estimates = preds.norm_estimates(S);
pred_names = preds.pred_names(S);
n_predictors = length(pred_names);
[~, idx] =sort(norm_estimates, 'descend');
best_predictors = pred_names(idx(1:n_predictors));
best_pred_vals = norm_estimates(idx(1:n_predictors));
end