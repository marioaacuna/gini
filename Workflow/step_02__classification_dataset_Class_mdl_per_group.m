clc
clear
global GC

% save filename
T_filename = fullfile(GC.raw_data_folder,'out', ['Re-organization_Mdl.xlsx']);

%% Read input table
table_filename = fullfile(GC.raw_data_folder,'in', "input.xlsx");
GC.variables_to_discard = {'Label', 'Experiment', 'ID', 'Slice', 'Date', 'SpikeCount', 'IInj'};

% Read table
T = readtable(table_filename);
opts.SelectedVariableNames = 'Label';

% get neurons only in L5
threshold = GC.threshold_depth_L5; % Thomas' paper
T(T.Depth < threshold, :) =[];

% make sure some nan values are set to 0 (not present)
T.Burst(isnan(T.Burst)) = 0;
T.ICAmp(isnan(T.ICAmp))= 0;

vars_to_eval = GC.variables_to_evaluate;
vars_to_eval = [vars_to_eval, 'ADP'];
T_pred = T(:,vars_to_eval);

%% Load Tree estimatiors
pred_filename = fullfile(GC.raw_data_folder, 'out', 'TREE_predictors.mat');
preds = load_variable(pred_filename, 'TREE_predictors');
norm_estimates = preds.norm_estimates;
pred_names = preds.pred_names;
n_predictors = 5;
[~, idx] =sort(norm_estimates, 'descend');
best_predictors = pred_names(idx(1:n_predictors));
best_pred_vals = norm_estimates(idx(1:n_predictors));

%% Trim table
% data_to_use_for_index = T_pred(:, best_predictors);

%% Load NaiveBayes model
mdl_filename = fullfile(GC.raw_data_folder, 'out','mdl_NaiveBayes_5_preds.mat');
mdl = load_variable(mdl_filename, 'mdl');

%% loop throug all conditions
Experiments = unique(T.Experiment);
Experiments(ismember(Experiments, {'Experiment', ''})) = []; 
T_all = []; % to be used for stats and visualizations
for iex = 1:numel(Experiments)
    this_experiment = Experiments{iex};
    this_T = T(ismember(T.Experiment, this_experiment),:);

    % delete nans
    this_array = table2array(this_T(:, best_predictors));
    isnan_idx = sum(isnan(this_array), 2) >1;
    d = this_array(~isnan_idx,:);
    
    % do PCA
    d_z= zscore(d);

    pca_d = do_pca_gini(d_z, best_pred_vals);
    y = mdl.predictFcn(pca_d);

    T_to_write = [this_T(~isnan_idx,:), table(y, 'VariableNames',{'Mdl_predictors'})];

    writetable(T_to_write, T_filename, "Sheet",this_experiment, 'WriteMode','overwritesheet')
    
    T_all = [T_all;T_to_write];
    figure
    gscatter(pca_d(:,1), pca_d(:,2), y);
    title(this_experiment)

end

fprintf('done writing in %s\n\n ', T_filename)

%% Check quick stats for neurons 1

r = 1;
D = table2array(T_all(:,best_predictors));
is_nan_idx_D = sum(isnan(D),2) >0;
D(is_nan_idx_D,:) = [];
D_z = zscore(D);
L = T_all.Mdl_predictors;
L(is_nan_idx_D) = [];
pcaD = do_pca_gini(D_z, best_pred_vals);

figure, gscatter(pcaD(:,1), pcaD(:,2), L);

%% check quickly stats for AP_thr
to_take = 'InputR';%'APThreshold'; 'SpikeCount'
data_groups = {'CFA d7NS', 'Saline d7NS'};
T_group = T(ismember(T.Experiment, data_groups),:);
data_both = T_group.APThreshold;
is_1 = T_all.Mdl_predictors ==1;
is_cfa = (ismember(T_all.Experiment, data_groups{1}) & is_1);
is_sal = (ismember(T_all.Experiment, data_groups{2})& is_1);

data_cfa = T_all.APThreshold(is_cfa);
data_saline = T_all.APThreshold(is_sal);

[~, p] = ttest2(data_cfa, data_saline)




%% check with sag
is_sag = T_all.SAG >= 1.1;
is_cfa = (ismember(T_all.Experiment, data_groups{1}) & is_sag);
is_sal = (ismember(T_all.Experiment, data_groups{2})& is_sag);
data_cfa = T_all.APThreshold(is_cfa);
data_saline = T_all.APThreshold(is_sal);


%% check sag from original 
is_sag = T.SAG >= 1.127;
is_cfa = (ismember(T.Experiment, data_groups{1}) & is_sag);
is_sal = (ismember(T.Experiment, data_groups{2})& is_sag);
data_cfa = T.(to_take)(is_cfa);
data_saline = T.(to_take)(is_sal);


