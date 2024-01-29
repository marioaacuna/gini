% read table
global GC
GC = general_configs();
% read table
table_filename = fullfile(GC.raw_data_folder, 'in','dlPAG_vs_vlPAG_vs_CC.xlsx');
T = readtable(table_filename);

% get neurons only in L5
threshold = GC.threshold_depth_L5; % Thomas' paper
T(T.Depth < threshold, :) =[];
T_new = T;

% make sure some nan values are set to 0 (not present)
T.Burst(isnan(T.Burst)) = 0;
T.ICAmp(isnan(T.ICAmp))= 0;

%% Load Tree estimatiors
[pred_names, best_predictors, best_pred_vals, idx] = load_best_predictors();

%% Trim table
data_to_use_for_index = T(:, best_predictors);
data_to_pca = table2array(data_to_use_for_index);

% clean nan values
% impute using KNN
%imputed = knnimpute(data_to_pca');
%imputed = imputed';
% % write down the number back to the table
% this_T = [not_best,array2table(imputed,'VariableNames',best_predictors)];
% d = imputed;

% alternatively, clean rows of data_to_pca containing nan values
isnan_idx = any(isnan(data_to_pca),2);
data_to_pca(any(isnan(data_to_pca),2),:) = [];
d = data_to_pca;

% define labels excluding isnan_idx
labels = T_new.Label;
labels(isnan_idx,:) = [];


%data_to_pca(isnan(data_to_pca)) = 0;

d_z = zscore(d);
% do PCA
% d_z= zscore(d);
% dz = (d-nanmean(d)) ./ nanstd(d);
pca_d = do_pca_gini(d_z, best_pred_vals);

% plot the pca in scatter plot
%figure;
%scatter(pca_d(:,1), pca_d(:,2), 50, 'filled');

%plot a gscatter with the labels
figure;
gscatter(pca_d(:,1), pca_d(:,2), labels);
legend({'CC', 'dlPAG', 'vlPAG'});

%% Eval model
% Load model
mdl_filename = fullfile(GC.raw_data_folder, 'out','mdl_LR_Signif_preds.mat');
mdl = load_variable(mdl_filename, 'mdl');
new_predictions = mdl.predictFcn(pca_d);

% plot a gscatter with the labels
figure
gscatter(pca_d(:,1), pca_d(:,2), new_predictions, [], 'o', 12)
hold on
gscatter(pca_d(:,1), pca_d(:,2),double(labels>0), [], 'x', 12)
% calculate accuracy
acc = sum(new_predictions ==  double(labels>0)) / length(labels);
%hold off
title(['Accuracy: ', num2str(acc*100), '%'])   

% id_dis = find(new_predictions ~=  double(labels>0));
% gscatter(pca_d(id_dis,1), pca_d(id_dis,2),double(labels(id_dis)>0), [], '.', 12)
legend({'IT', 'SC'});
%legend({'real', 'predicted'});
hold off

% find repeated values in pca_d
[~, idx] = unique(pca_d, 'rows', 'stable');

