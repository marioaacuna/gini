% This script is designed to evaluate neurons that were collected by
% tracing - so far from the cACC and from the PAG (dl and vl).
% We first calculate an idex, based on the most important features. Which
% is then used for the threshold calculation
% 
%   - compute the feature importance based on fit tree 
%   - Compute the Index values and set threshold
%   * So far we only use this threshold for the classification (saved in 
%   the predictor structure). Tjhis is later passed to evaluate new cells
%   This is still in beta version. Additionally:
%   - compute PCA form the data set.
%
% Rootfolder: C:\Users\acuna\Documents\Gini
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc
close all

%% Init variables
save_figs = 0 ;
FP = figure_properties();
type_experiment = 'retro';

color_a = FP.colors.groups.a;
color_b = FP.colors.groups.b;

global GC
GC = general_configs();
% read table
table_filename = fullfile(GC.raw_data_folder, 'Retro_ACC_PAG.xlsx');
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
vars_to_eval = [vars_to_eval, 'ADP'];
T_use = T_new(:,vars_to_eval);

%% Run a tree classifier to determine the important variables
tree_1 = fitrtree(T_new(:,ismember(T_new.Properties.VariableNames, vars_to_eval)), response,'PredictorSelection','curvature','Surrogate','on');
% Plot figure importance
imp = predictorImportance(tree_1);
figure_predictors = figure('Color','w');
[imp_sorted, sorted_idx] = sort(imp, 'ascend');
is_zero = imp_sorted==0;
estimates = imp_sorted(~is_zero);
norm_estimates = estimates / max(estimates);
barh(norm_estimates);
title('Predictor Importance Estimates');
xlabel('Estimates');
ylabel('Predictors');
h = gca;
pred_names = tree_1.PredictorNames(sorted_idx);
pred_names = pred_names(~is_zero);
yticks([1:length( pred_names)])
h.YTickLabel = pred_names;
% h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
box off
toggle_toolbox('_plotting', 'on')
fig_filename = os.path.join(GC.plot_path, type_experiment,'fig_predictors_retro_PAG.pdf');
%% save fig
% export_fig(fig_filename, '-pdf', '-q101', '-nocrop', '-painters',figure_predictors); close(figure_predictors)

%% Get BEST predictors
n_predictors = 3;
[~, idx] =sort(norm_estimates, 'descend');

% take the 2-3 highest predictors
best_predictors = pred_names(idx(1:n_predictors));
best_pred_vals = norm_estimates(idx(1:n_predictors));
data_to_use_for_index = T_new(:, best_predictors);

% Alternative 1 : weighted sum approach
%Index = w1 * MaxV_1 + w2 * MaxOrder + w3 * SAG
pred = [];
for ip = 1:n_predictors
    pred(:,ip) = best_pred_vals(ip)* table2array(data_to_use_for_index(:,best_predictors(:,ip)));
end
Indexes = sum(pred, 2);
% Indexes = best_pred_vals(:,1)* table2array(data_to_use_for_index(:,best_predictors(:,1))) + ...
%         best_pred_vals(:,2)* table2array(data_to_use_for_index(:,best_predictors(:,2))) + ...
%         best_pred_vals(:,3)* table2array(data_to_use_for_index(:,best_predictors(:,3)));
figure, gscatter(Indexes, repmat(1, size(Indexes)), response, [], 'o')

% Calculate the ROC from indexes
% Calculate true positive rate (TPR) and false positive rate (FPR) for different threshold values
nThresholds = 100; % Number of threshold values to evaluate
thresholds = linspace(min(Indexes), max(Indexes), nThresholds); % Generate a range of threshold values
idx2 = Indexes((response==1));
idx1 = Indexes((response==0));
TPR = zeros(nThresholds,1); % Initialize TPR vector
FPR = zeros(nThresholds,1); % Initialize FPR vector
for i = 1:nThresholds
    TP = sum(idx2 >= thresholds(i)); % Number of true positives
    FN = sum(idx2 < thresholds(i)); % Number of false negatives
    FP = sum(idx1 >= thresholds(i)); % Number of false positives
    TN = sum(idx1 < thresholds(i)); % Number of true negatives
    TPR(i) = TP / (TP + FN); % True positive rate
    FPR(i) = FP / (FP + TN); % False positive rate
end

% Calculate area under ROC curve
AUC = trapz(FPR, TPR); % Area under ROC curve

% Plot ROC curve
figure;
plot(FPR, TPR, 'b-', 'LineWidth', 2);
xlabel('False positive rate');
ylabel('True positive rate');
title(sprintf('ROC curve (AUC = %.3f)', AUC));
axis square;

% In this example, we assume that the two populations have index values that 
% follow normal distributions with different means and variances, but you can 
% substitute your own data. The code calculates the TPR and FPR for different 
% threshold values using a loop, then plots the ROC curve using the plot function.
% Finally, it calculates the area under the curve using the trapz function and 
% displays it in the plot title.

% Find threshold that maximizes separation based on ROC curve
ideal_point = [0, 1]; % Ideal point on ROC curve
distances = sqrt(sum((ideal_point - [FPR, TPR]).^2, 2)); % Euclidean distances from each point on ROC curve to ideal point
[min_distance, min_idx] = min(distances); % Index of point on ROC curve closest to ideal point
threshold_max_sep = thresholds(min_idx); % Threshold that maximizes separation

THR_AUROC = threshold_max_sep;

% save predictors and threshold
predictors_filename = fullfile(GC.raw_data_folder, 'out', 'predictor_weights.mat');
predictors = struct();
predictors.weights = best_pred_vals;
predictors.names = best_predictors;
predictors.threshold = THR_AUROC;
save(predictors_filename, 'predictors'); disp('Predictors saved')

% Plot form indexes
fpca = figure('Color','w');
gscatter(Indexes, repmat(1, size(Indexes,1)), response, [], 'o')
set(gca,'TickDir','out', 'box', 'off');
title(['Using indexes first ', num2str(n_predictors),' components'])
xlabel('Index')
hold on
plot([THR_AUROC,THR_AUROC   ], [0,2], '--')
hold off

fig_filename = os.path.join(GC.plot_path, type_experiment,'fig_INDEX_predictors_retro_PAG.pdf');
% save fig
%export_fig(fig_filename, '-pdf', '-q101', '-nocrop', '-painters',fpca); close(fpca)


%% PLOT PCA all predictors 
n_predictors = 16;
[~, idx] =sort(norm_estimates, 'descend');

% take the 2-3 highest predictors
best_predictors = pred_names(idx(1:n_predictors));
best_pred_vals = norm_estimates(idx(1:n_predictors));
data_to_use_for_index = T_new(:, best_predictors);

d = table2array(data_to_use_for_index);
d_z= zscore(d);
pca_d = pca(d_z', "NumComponents",2, 'Algorithm','svd', 'Centered',false,'Weights', best_pred_vals);
fpca = figure;
gscatter(pca_d(:,1), pca_d(:,2), response)
set(gca,'TickDir','out', 'box', 'off');
title('pca')
xlabel('PC1')
ylabel('PC2')

fig_filename = os.path.join(GC.plot_path, type_experiment,'fig_PCA_all_predictors_retro_PAG.pdf');
% save fig
%export_fig(fig_filename, '-pdf', '-q101', '-nocrop', '-painters',fpca); close(fpca)


%% run pca to determine threshold -> Not necessary anymore
% %z_new= zscore(table2array(T_use), 1, 2);
% best_predictos_idx = find(ismember(T_new.Properties.VariableNames, best_predictors));%
% % z_new= zscore(table2array(T_new(:,best_predictors)), 0,2);
% [pca_eval] = pca(table2array(T_new(:,best_predictors))', 'NumComponents',1, 'Algorithm','svd', 'Centered',false, 'Weights', best_pred_vals);
% fpca = figure('Color','w');
% gscatter(pca_eval(:,1), repmat(1, size(pca_eval,1)), response, [], 'o')
% set(gca,'TickDir','out', 'box', 'off');
% title('pca')
% xlabel('PC1')
% % find threshold
% min_b =min(pca_eval(response==1));
% max_a =max(pca_eval(response==0));
% THR_mean = mean([min_b, max_a]);
% hold on
% plot([THR_mean,THR_mean], [0,2], '--')
% hold off
% 
% % generate thr based on mean and std from data
% lower_1 = mean(pca_eval(response==1)) - 2.5*std(pca_eval(response==1))/2;
% upper_0 = mean(pca_eval(response==0)) + 2.5*std(pca_eval(response==0))/2;
% 
% 
% fig_filename = os.path.join(GC.plot_path, type_experiment,'fig_PCA_predictors_retro_PAG.pdf');
% % save fig
% export_fig(fig_filename, '-pdf', '-q101', '-nocrop', '-painters',fpca); close(fpca)




%% Calculate the ROC from pca
% Calculate true positive rate (TPR) and false positive rate (FPR) for different threshold values
nThresholds = 100; % Number of threshold values to evaluate
thresholds = linspace(min(pca_eval), max(pca_eval), nThresholds); % Generate a range of threshold values
idx2 = pca_eval((response==1));
idx1 = pca_eval((response==0));
TPR = zeros(nThresholds,1); % Initialize TPR vector
FPR = zeros(nThresholds,1); % Initialize FPR vector
for i = 1:nThresholds
    TP = sum(idx2 >= thresholds(i)); % Number of true positives
    FN = sum(idx2 < thresholds(i)); % Number of false negatives
    FP = sum(idx1 >= thresholds(i)); % Number of false positives
    TN = sum(idx1 < thresholds(i)); % Number of true negatives
    TPR(i) = TP / (TP + FN); % True positive rate
    FPR(i) = FP / (FP + TN); % False positive rate
end

% Calculate area under ROC curve
AUC = trapz(FPR, TPR); % Area under ROC curve

% Plot ROC curve
figure;
plot(FPR, TPR, 'b-', 'LineWidth', 2);
xlabel('False positive rate');
ylabel('True positive rate');
title(sprintf('ROC curve (AUC = %.3f)', AUC));
axis square;

% In this example, we assume that the two populations have index values that 
% follow normal distributions with different means and variances, but you can 
% substitute your own data. The code calculates the TPR and FPR for different 
% threshold values using a loop, then plots the ROC curve using the plot function.
% Finally, it calculates the area under the curve using the trapz function and 
% displays it in the plot title.

% Find threshold that maximizes separation based on ROC curve
ideal_point = [0, 1]; % Ideal point on ROC curve
distances = sqrt(sum((ideal_point - [FPR, TPR]).^2, 2)); % Euclidean distances from each point on ROC curve to ideal point
[min_distance, min_idx] = min(distances); % Index of point on ROC curve closest to ideal point
threshold_max_sep = thresholds(min_idx); % Threshold that maximizes separation

THR_AUROC = threshold_max_sep;
%% CALCULATE AUROC FROM SAG
% Calculate the ROC
% Calculate true positive rate (TPR) and false positive rate (FPR) for different threshold values
nThresholds = 100; % Number of threshold values to evaluate
thresholds = linspace(min(T_new.SAG), max(T_new.SAG), nThresholds); % Generate a range of threshold values
idx2 = T_new.SAG((response==1));
idx1 = T_new.SAG((response==0));
TPR = zeros(nThresholds,1); % Initialize TPR vector
FPR = zeros(nThresholds,1); % Initialize FPR vector
for i = 1:nThresholds
    TP = sum(idx2 >= thresholds(i)); % Number of true positives
    FN = sum(idx2 < thresholds(i)); % Number of false negatives
    FP = sum(idx1 >= thresholds(i)); % Number of false positives
    TN = sum(idx1 < thresholds(i)); % Number of true negatives
    TPR(i) = TP / (TP + FN); % True positive rate
    FPR(i) = FP / (FP + TN); % False positive rate
end

% Calculate area under ROC curve
AUC = trapz(FPR, TPR); % Area under ROC curve

% Plot ROC curve
figure;
plot(FPR, TPR, 'b-', 'LineWidth', 2);
xlabel('False positive rate');
ylabel('True positive rate');
title(sprintf('ROC curve (AUC = %.3f)', AUC));
axis square;

% In this example, we assume that the two populations have index values that 
% follow normal distributions with different means and variances, but you can 
% substitute your own data. The code calculates the TPR and FPR for different 
% threshold values using a loop, then plots the ROC curve using the plot function.
% Finally, it calculates the area under the curve using the trapz function and 
% displays it in the plot title.

% Find threshold that maximizes separation based on ROC curve
ideal_point = [0, 1]; % Ideal point on ROC curve
distances = sqrt(sum((ideal_point - [FPR, TPR]).^2, 2)); % Euclidean distances from each point on ROC curve to ideal point
[min_distance, min_idx] = min(distances); % Index of point on ROC curve closest to ideal point
threshold_max_sep = thresholds(min_idx); % Threshold that maximizes separation

THR_AUROC_SAG = threshold_max_sep;

% % save predictors and threshold
% predictors_filename = fullfile(GC.raw_data_folder, 'out', 'predictor_weights.mat');
% predictors = struct();
% predictors.wights = best_pred_vals;
% predictors.names = best_predictors;
% predictors.threshold = THR_AUROC;
% save(predictors_filename, 'predictors')
% 







%% Plot the thrshoold calculated by AUROC for SAG
fpca = figure('Color','w');
gscatter(T_new.SAG, repmat(1, size(T_new.SAG,1)), response, [], 'o')
set(gca,'TickDir','out', 'box', 'off');
title('Using SAG')
xlabel('PC1')
hold on
plot([THR_AUROC_SAG,THR_AUROC_SAG   ], [0,2], '--')
hold off


% %% analyse Blind-obtained neurons
% input_file = os.path.join(GC.raw_data_folder, 'Only_saline_data.xlsx');
% opts = detectImportOptions(input_file);
% parameters = opts.VariableNames;
% % variables_to_discard = GC.variables_to_discard;%{'Date', 'Slice', 'ID', 'Burst'}; % , 'Burst', 'ICAmp'
% % Variables = parameters(~ismember(parameters, variables_to_discard)); 
% % Variables = parameters(ismember(parameters, {'SAG', 'Diameter'}));
% opts.SelectedVariableNames = parameters;
% % Read table
% T_eval_ori=readtable(input_file);
% opts.SelectedVariableNames = 'Label';
% T_eval_ori(T_eval_ori.Depth < threshold, :) =[];
% % Set nan values to 0
% T_eval_ori.ICAmp(isnan(T_eval_ori.ICAmp))= 0;
% try
%     T_eval_ori.Burst(isnan(T_eval_ori.Burst)) = 0;
% catch
%     keyboard
% end
% %T(:,[1,2]) = [];
% 
% % vars_new = opts.VariableNames(:,~ismember(opts.VariableNames,{'Label','Date', 'Experiment', 'Slice', 'ID', 'IInj', 'SpikeCount'}));
% % check if the new data has also the same variables
% % vars_intersectt = intersect(vars_to_eval, opts.VariableNames);
% try
%     T_eval = T_eval_ori(:,best_predictors);
% catch
%     keyboard
% end
% % vars_to_eval = intersect(vars_new  , T_new.Properties.VariableNames);
% pred_T_eval = T_eval_ori.Label;
% pred_T_eval = double(ismember(pred_T_eval  , 'b'));
% %T_eval = [T_eval, table(pred_T_eval, 'VariableNames',{'response'})]
% % delete nan values
% isnan_idx = (sum(isnan(table2array(T_eval)), 2)>0);
% T_eval(isnan_idx,:) = [];
% pred_T_eval(isnan_idx) =[];


%% obtain labels using PCA thresholding
% z_eval = zscore(table2array(T_eval(:,best_predictors)));
% [pca_eval] = pca(z_eval', 'NumComponents',1, 'Algorithm','svd', 'Centered',false, 'Weights', best_pred_vals);
% fpca = figure('Color','w');
% gscatter(pca_eval(:,1), repmat(1, size(pca_eval,1)), pred_T_eval, [], 'o')
% set(gca,'TickDir','out', 'box', 'off');
% title('pca on blind data')
% xlabel('PC1')
% hold on
% plot([THR,THR], [0,2], '--')
% hold off
% %% Lower and upper bounds
% fpca = figure('Color','w');
% gscatter(pca_eval(:,1), repmat(1, size(pca_eval,1)), pred_T_eval, [], 'o')
% set(gca,'TickDir','out', 'box', 'off');
% title('pca on blind data')
% xlabel('PC1')
% hold on
% plot([lower_1,lower_1], [0,2], '--r')
% plot([upper_0,upper_0], [0,2], '--b')
% hold off















%% 
% T_use = T_new(:,vars_to_eval);
% T_use = T_new(:,pred_names)
% %% Train a model, or load pre-trained model
% % use the GUI
% %keyboard
% % save model
% %save(fullfile(GC.raw_data_folder, 'NaiveBayes_MDL_23_3_classes.mat'), 'mdl_23_3_classes')
% %save(fullfile(GC.raw_data_folder, 'NaiveBayes_MDL_25_features.mat'), 'mdl')
% % load model
% mdl = load_variable(fullfile(GC.raw_data_folder,'models', 'NaiveBayes_MDL_24.mat'), 'mdl_24'); % NaiveBayes_MDL_24; NaiveBayes_MDL_23_3_classes
% %mdl = load_variable(fullfile(GC.raw_data_folder,'models', 'NaiveBayes_MDL_23_3_classes.mat'), 'mdl_23_3_classes'); % NaiveBayes_MDL_24; NaiveBayes_MDL_23_3_classes
% %% Predict
% [yfit,~] = mdl.predictFcn(T_eval);
% %%
% figure
% c = confusionchart(pred_T_eval, yfit,"Normalization","row-normalized");
% acc = confusion_matrix_stats(c.NormalizedValues); acc=acc.avg_accuracy;
% title(['ACC: ', num2str(acc)])
% 
% % run pca
% pca_eval = pca(table2array(T_eval)', 'NumComponents',2, 'Weight', best_pred_vals);
% figure, 
% gscatter(pca_eval(:,1), pca_eval(:,2), yfit)
% yfit(isnan_idx) = [];
% tsne_data = tsne(table2array(T_eval), 'Algorithm','exact', 'Distance','correlation',...
% 'NumDimensions',2, 'NumPCAComponents',2, ...
% 'Standardize',1, 'Verbose',0, 'Exaggeration',4, 'Perplexity',5, 'LearnRate', 500);
% 

%% Run predictions and pca in retro data
[yfit,~] = mdl.predictFcn(T_use);
%%
% figure
% c = confusionchart(response, yfit,"Normalization","row-normalized");
% acc = confusion_matrix_stats(c.NormalizedValues); acc=acc.avg_accuracy;
% title(['ACC: ', num2str(acc)])
zdata = zscore(table2array(T_new(:,pred_names))); 
% run pca
[ pca_eval] = pca(zdata', 'NumComponents',2, 'Algorithm','svd', 'Centered',false, 'Weights', norm_estimates);
f = figure('Color','w');
gscatter(pca_eval(:,1), pca_eval(:,2), response, [], 'o')
set(gca,'TickDir','out', 'box', 'off');
title('pca')
xlabel('PC1')
ylabel('PC2')
fig_filename = os.path.join(GC.plot_path, 'S','fig_PCA_retro_PAG.pdf');
%% save fig
export_fig(fig_filename, '-pdf', '-q101', '-nocrop', '-painters',f); close(f)

% %% Run Tsne on retro data
% tsne_data = tsne(zdata, 'Algorithm','exact', 'Distance','correlation',...
% 'NumDimensions',2, 'NumPCAComponents',2, ...
% 'Standardize',0, 'Verbose',0, 'Exaggeration',2, 'Perplexity',2, 'LearnRate', 1000);
% figure, 
% gscatter(tsne_data(:,1), tsne_data(:,2),response)
% title('tsne')







