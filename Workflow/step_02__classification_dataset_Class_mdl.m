clc
clear
global GC
%% read retro table
table_filename = fullfile(GC.raw_data_folder, 'in','Retro_ACC_PAG.xlsx');
opts = detectImportOptions(table_filename);
parameters = opts.VariableNames;
% GC.variables_to_discard = {'Label', 'Experiment', 'ID', 'Slice', 'Date', 'SpikeCount', 'IInj'};
variables_to_discard = GC.variables_to_discard;%{'Date', 'Slice', 'ID', 'Burst'}; % , 'Burst', 'ICAmp'

% Variables = parameters(~ismember(parameters, variables_to_discard)); 
% opts.SelectedVariableNames = Variables;
%T = readtable( table_filename,opts);

% Read table
T = readtable(table_filename);
opts.SelectedVariableNames = 'Label';

% get neurons only in L5
threshold = GC.threshold_depth_L5; % Thomas' paper
T(T.Depth < threshold, :) =[];
T_new = T;
% make sure some nan values are set to 0 (not present)
T_new.ICAmp(isnan(T_new.ICAmp))= 0;
T_new.Burst(isnan(T_new.Burst)) = 0;
original_labels = T_new.Label;
% actual labels for REtroACCPAG data are [0,1] instead of [a, b].
response = double(ismember(original_labels, 1));
vars_to_eval = GC.variables_to_evaluate;
% delete burst

vars_to_eval = [vars_to_eval, 'ADP'];
T_train = T_new(:,vars_to_eval);

%% Read input table

table_filename = fullfile(GC.raw_data_folder,'in', "input.xlsx");
GC.variables_to_discard = {'Label', 'Experiment', 'ID', 'Slice', 'Date', 'SpikeCount', 'IInj'};

% Variables = parameters(~ismember(parameters, variables_to_discard)); 
% opts.SelectedVariableNames = Variables;
%T = readtable( table_filename,opts);

% Read table
T = readtable(table_filename);
opts.SelectedVariableNames = 'Label';

% get neurons only in L5
threshold = GC.threshold_depth_L5; % Thomas' paper
T(T.Depth < threshold, :) =[];

% make sure some nan values are set to 0 (not present)
T.Burst(isnan(T.Burst)) = 0;
T.ICAmp(isnan(T.ICAmp))= 0;
pred = T;
%% Run tree
tree_1 = fitrtree(T_train(:,ismember(T_train.Properties.VariableNames, vars_to_eval)), response,'PredictorSelection','curvature','Surrogate','on');
% Plot figure importance
imp = predictorImportance(tree_1);
[imp_sorted, sorted_idx] = sort(imp, 'ascend');
is_zero = imp_sorted==0;
estimates = imp_sorted(~is_zero);
norm_estimates = estimates / max(estimates);

%% Get BEST predictors
n_predictors = 5;
% n_predictors =length(norm_estimates);
[~, idx] =sort(norm_estimates, 'descend');
pred_names = tree_1.PredictorNames(sorted_idx);
pred_names = pred_names(~is_zero);

% take the 2-3 highest predictors
best_predictors = pred_names(idx(1:n_predictors));
best_pred_vals = norm_estimates(idx(1:n_predictors));


%% Compute PCA on training data
data_to_use_for_index = T_train(:, best_predictors);

d = table2array(data_to_use_for_index);
d_z= zscore(d);
pca_d = pca(d_z', "NumComponents",2, 'Algorithm','svd', 'Centered',false,'Weights', best_pred_vals);




%% model:
% from app : model -> Naive Bayes
mdl = trainedModel;
% from app : model -> SVM
mdl2 = trainedModel1;



%%

T_pred_to_use = T_pred;
pred= table2array(T_pred_to_use(:, best_predictors));

is_nan_idx = sum(isnan(pred),2) >0;
pred = pred(~is_nan_idx,:);
pred_z = zscore(pred);
pred_pca = pca(pred_z', "NumComponents",2, 'Algorithm','svd', 'Centered',false,'Weights', best_pred_vals);
figure, scatter(pred_pca(:,1), pred_pca(:,2))


%% predict with model
[yfit_nb,~] = mdl.predictFcn(pred_pca);
figure, gscatter(pred_pca(:,1), pred_pca(:,2), yfit_nb, [],'o', 'filled')
title('Naive Bayes')
[yfit_svm,~] = mdl2.predictFcn(pred_pca);
figure, gscatter(pred_pca(:,1), pred_pca(:,2), yfit_svm, [],'o', 'filled')
title('SVM')
%%


%% write table NB
T_to_write = [T(~is_nan_idx,:), table(yfit_nb, 'VariableNames',{'LabelPrediction'})];
new_table_filename = fullfile(GC.raw_data_folder,'out',"input_with_predicted_lables_NaiveBayes_5preds.xlsx");
if exist("new_table_filename", 'file')
    disp('it will re-write')
end
writetable(T_to_write, new_table_filename)


%% Write table SVM
T_to_write = [T(~is_nan_idx,:), table(yfit_svm, 'VariableNames',{'LabelPrediction'})];
new_table_filename = fullfile(GC.raw_data_folder,'out',"input_with_predicted_lables_SVM.xlsx");
if exist("new_table_filename", 'file')
    disp('it will re-write')
end
writetable(T_to_write, new_table_filename)

