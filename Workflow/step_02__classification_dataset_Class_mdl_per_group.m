clc
clear
global GC
addpath('Code\Utilities\Statistics')
% save filename
T_filename = fullfile(GC.raw_data_folder,'out', ['Re-organization_Mdl_LR_Signif_preds.xlsx']);

%% Read input table
table_filename = fullfile(GC.raw_data_folder,'in', "input.xlsx");
GC.variables_to_discard = {'Label', 'Experiment', 'ID', 'Slice', 'Date', 'SpikeCount', 'IInj'};

% Read table
T = readtable(table_filename);
opts.SelectedVariableNames = 'Label';

% get neurons only in L5
threshold = GC.threshold_depth_L5; % Thomas' paper
% threshold = 350;
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
S = preds.S;
norm_estimates = preds.norm_estimates(S);
pred_names = preds.pred_names(S);
n_predictors = length(pred_names);
[~, idx] =sort(norm_estimates, 'descend');
best_predictors = pred_names(idx(1:n_predictors));
best_pred_vals = norm_estimates(idx(1:n_predictors));

%% Trim table
% data_to_use_for_index = T_pred(:, best_predictors);

%% Load NaiveBayes model
mdl_filename = fullfile(GC.raw_data_folder, 'out','mdl_LR_Signif_preds.mat');
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
    not_best = this_T(:,~ismember(this_T.Properties.VariableNames, best_predictors));

    % impute using KNN
    imputed = knnimpute(this_array);
    % write down the number back to the table
    this_T = [not_best,array2table(imputed,'VariableNames',best_predictors)];
    d = imputed;
    % isnan_idx = sum(isnan(this_array), 2) >0;
    % d = this_array(~isnan_idx,:);
    
    % do PCA
    d_z= zscore(d);
    % dz = (d-nanmean(d)) ./ nanstd(d);
    pca_d = do_pca_gini(d_z, best_pred_vals);
    y = mdl.predictFcn(pca_d);

    % T_to_write = [this_T(~isnan_idx,:), table(y, 'VariableNames',{'Mdl_predictors'})];
    T_to_write = [this_T, table(y, 'VariableNames',{'Mdl_predictors'})];

    writetable(T_to_write, T_filename, "Sheet",this_experiment, 'WriteMode','overwritesheet')
    
    T_all = [T_all;T_to_write];
    figure
    gscatter(pca_d(:,1), pca_d(:,2), y);
    title(this_experiment)

end


%% Write Reorganized table

step_03__fig2_reorganizedata2excel_from_Mdl(T_all)

fprintf('done writing in %s\n\n ', T_filename)
%%

%% Check quick stats for neurons 1

r = 1;
TD = (T_all(sum(isnan(table2array(T_all(:, vars_to_eval))),2) ==0,[best_predictors,'Experiment', 'Mdl_predictors', 'APThreshold']));
% is_nan_idx_D = sum(isnan(D),2) >0;
% D(is_nan_idx_D,:) = [];
D = table2array(TD(:, best_predictors));
D_z = zscore(D);
pcaD = do_pca_gini(D_z, best_pred_vals);

L = mdl.predictFcn(pcaD);
% L = T_all.Mdl_predictors;


figure, gscatter(pcaD(:,1), pcaD(:,2), L);

%% check quickly stats for AP_thr
to_take = 'SpikeCount';%'APThreshold'; 'SpikeCount'
data_groups = {'CFA d7NS', 'Saline d7NS'};


TD_all = (T_all(:,[best_predictors,'Experiment', 'Mdl_predictors', 'APThreshold']));
T_group = TD_all(ismember(TD_all.Experiment, data_groups),:);
is_1 = TD_all.Mdl_predictors == 1;
is_cfa = (ismember(TD_all.Experiment, data_groups{1}) & is_1);
is_sal = (ismember(TD_all.Experiment, data_groups{2})& is_1);

data_cfa = TD_all.(to_take)(is_cfa);
data_saline = TD_all.(to_take)(is_sal);

[~, p] = ttest2(data_cfa, data_saline)
%%

zTD = zscore(table2array(T_all(:, best_predictors)))
pcaTD = do_pca_gini(zTD, best_pred_vals);
figure, gscatter(pcaTD(:,1), pcaTD(:,2), is_1);

% LR_Sig_pred per group yielded significance for InputR but not (almost) for APthr


%% check with sag
is_sag = T.SAG >= 1.1;
is_cfa = (ismember(T.Experiment, data_groups{1}) & is_sag);
is_sal = (ismember(T.Experiment, data_groups{2})& is_sag);
data_cfa = T.APThreshold(is_cfa);
data_saline = T.APThreshold(is_sal);


%% check sag from original 
is_sag = T.SAG >= 1.127;
is_cfa = (ismember(T.Experiment, data_groups{1}) & is_sag);
is_sal = (ismember(T.Experiment, data_groups{2})& is_sag);
data_cfa = T.(to_take)(is_cfa);
data_saline = T.(to_take)(is_sal);


