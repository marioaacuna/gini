% This script is designed to evaluate neurons that were collected by
% tracing - so far from the cACC and from the PAG (dl and vl).
% if no model has been created will load a model trained with this data and
% test in a separete data set (altho this is still in beta version).
%   - compute the feature importance based on fit tree 
%   - compute PCA form the data set.
%   - Plot all possible parameters
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
vars_to_eval = opts.VariableNames(:,~ismember(T_new.Properties.VariableNames, {'Label', 'Experiment', 'ID', 'Slice', 'Date', 'SpikeCount', 'IInj'}));

T_use = T_new(:,vars_to_eval);

%% Run a tree classifier to determine the important variables
tree_1 = fitrtree(T_new(:,~ismember(T_new.Properties.VariableNames, {'Label', 'Experiment', 'ID', 'Slice', 'Date', 'SpikeCount', 'IInj'})), response,'PredictorSelection','curvature','Surrogate','on');
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
export_fig(fig_filename, '-pdf', '-q101', '-nocrop', '-painters',figure_predictors); close(figure_predictors)

%% Main predictors 
% take the 3 highest predictors
best_predictors = pred_names(end-2:end);
best_pred_vals = norm_estimates(end-2:end);
data_to_use_for_index = T_new(:, best_predictors);
% Alternative 1 : weighted sum approach
%Index = w1 * MaxV_1 + w2 * MaxOrder + w3 * SAG
Indexes = best_pred_vals(:,1)* table2array(data_to_use_for_index(:,best_predictors(:,1))) + ...
        best_pred_vals(:,2)* table2array(data_to_use_for_index(:,best_predictors(:,2))) + ...
        best_pred_vals(:,3)* table2array(data_to_use_for_index(:,best_predictors(:,3)));

figure, gscatter(repmat(1, size(Indexes)), Indexes, response)


% run pca to determine threshold
z_new= zscore(table2array(T_new(:,best_predictors)));
[pca_eval] = pca(z_new', 'NumComponents',1, 'Algorithm','svd', 'Centered',false, 'Weights', best_pred_vals);
fpca = figure('Color','w');
gscatter(pca_eval(:,1), repmat(1, size(pca_eval,1)), response, [], 'o')
set(gca,'TickDir','out', 'box', 'off');
title('pca')
xlabel('PC1')
% find threshold
min_b =min(pca_eval(response==1));
max_a =max(pca_eval(response==0));
THR_mean = mean([min_b, max_a]);
hold on
plot([THR_mean,THR_mean], [0,2], '--')
hold off

% generate thr based on mean and std from data
lower_1 = mean(pca_eval(response==1)) - 2.5*std(pca_eval(response==1))/2;
upper_0 = mean(pca_eval(response==0)) + 2.5*std(pca_eval(response==0))/2;



%% Calculate the ROC
% Calculate true positive rate (TPR) and false positive rate (FPR) for different threshold values
nThresholds = 100; % Number of threshold values to evaluate
thresholds = linspace(min(pca_eval), max(pca_eval), nThresholds); % Generate a range of threshold values
idx1 = pca_eval((response==1));
idx2 = pca_eval((response==0));
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
predictors.wights = best_pred_vals;
predictors.names = best_predictors;
predictors.threshold = THR_AUROC;
save(predictors_filename, 'predictors')


%% analyse Blind-obtained neurons
input_file = os.path.join(GC.raw_data_folder, 'Only_saline_data.xlsx');
opts = detectImportOptions(input_file);
parameters = opts.VariableNames;
% variables_to_discard = GC.variables_to_discard;%{'Date', 'Slice', 'ID', 'Burst'}; % , 'Burst', 'ICAmp'
% Variables = parameters(~ismember(parameters, variables_to_discard)); 
% Variables = parameters(ismember(parameters, {'SAG', 'Diameter'}));
opts.SelectedVariableNames = parameters;
% Read table
T_eval_ori=readtable(input_file);
opts.SelectedVariableNames = 'Label';
T_eval_ori(T_eval_ori.Depth < threshold, :) =[];
% Set nan values to 0
T_eval_ori.ICAmp(isnan(T_eval_ori.ICAmp))= 0;
try
    T_eval_ori.Burst(isnan(T_eval_ori.Burst)) = 0;
catch
    keyboard
end
%T(:,[1,2]) = [];

% vars_new = opts.VariableNames(:,~ismember(opts.VariableNames,{'Label','Date', 'Experiment', 'Slice', 'ID', 'IInj', 'SpikeCount'}));
% check if the new data has also the same variables
% vars_intersectt = intersect(vars_to_eval, opts.VariableNames);
try
    T_eval = T_eval_ori(:,best_predictors);
catch
    keyboard
end
% vars_to_eval = intersect(vars_new  , T_new.Properties.VariableNames);
pred_T_eval = T_eval_ori.Label;
pred_T_eval = double(ismember(pred_T_eval  , 'b'));
%T_eval = [T_eval, table(pred_T_eval, 'VariableNames',{'response'})]
% delete nan values
isnan_idx = (sum(isnan(table2array(T_eval)), 2)>0);
T_eval(isnan_idx,:) = [];
pred_T_eval(isnan_idx) =[];


%% obtain labels using PCA thresholding
z_eval = zscore(table2array(T_eval(:,best_predictors)));
[pca_eval] = pca(z_eval', 'NumComponents',1, 'Algorithm','svd', 'Centered',false, 'Weights', best_pred_vals);
fpca = figure('Color','w');
gscatter(pca_eval(:,1), repmat(1, size(pca_eval,1)), pred_T_eval, [], 'o')
set(gca,'TickDir','out', 'box', 'off');
title('pca on blind data')
xlabel('PC1')
hold on
plot([THR,THR], [0,2], '--')
hold off
%% Lower and upper bounds
fpca = figure('Color','w');
gscatter(pca_eval(:,1), repmat(1, size(pca_eval,1)), pred_T_eval, [], 'o')
set(gca,'TickDir','out', 'box', 'off');
title('pca on blind data')
xlabel('PC1')
hold on
plot([lower_1,lower_1], [0,2], '--r')
plot([upper_0,upper_0], [0,2], '--b')
hold off















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

%% PLOTS
%%
is_b = response == 1;
is_a = response==0;

data = T_use;
x_basal_b = data.MaxH(is_b);
y_basal_b = data.MaxV(is_b);
x_basal_a = data.MaxH(is_a);
y_basal_a = data.MaxV(is_a);
depth_b = data.Depth(is_b);
depth_a = data.Depth(is_a);
y0_b = mean(data.Depth(is_b));
y0_a = mean(data.Depth(is_a));
%%
fig_all_areas = figure('color', 'w');
%
x0 = 0;
% y0 = y0_b;
x0a = 400;
for i_cell = 1: sum(is_b)
    y0 = depth_b(i_cell);
    a = x_basal_b(i_cell)/2;
    b = y_basal_b(i_cell)/2;
    t = linspace(-pi,pi,100);
    x = 1 - x0 + a*cos(t);
    y = 1 - y0 + b*sin(t);
    plot(x,y)
    hold on
end
% get the mean bifurcation point
bifurcation_b = T.Bifurcation(is_b);
% y1 = nanmean(bifurcation_b) - y0_b ;
x1 = 0;
x_apical_b = data.MaxH_1(is_b);
y_apical_b = data.MaxV_1(is_b);
Y = NaN(sum(is_b),1);
for i_cell = 1: sum(is_b)
    y1 = bifurcation_b(i_cell) + 1/2*y_apical_b(i_cell) - depth_b(i_cell);
    Y(i_cell) = y1;
    a = x_apical_b(i_cell)/2;
    b = y_apical_b(i_cell)/2;
    t = linspace(-pi,pi,100);
    x = 1 + x1 + a*cos(t);
    y = 1 + y1 + b*sin(t);
    plot(x,y)
    hold on
end

% plot a
bifurcation_a = T.Bifurcation(is_a);
x0 = 400;
% y0 = y0_a;
for i_cell = 1: sum(is_a)
    y0 = depth_a(i_cell);
    a = x_basal_a(i_cell)/2;
    b = y_basal_a(i_cell)/2;
    t = linspace(-pi,pi,100);
    x = 1 + x0 + a*cos(t);
    y = 1 - y0 + b*sin(t);
    plot(x,y)
    hold on
end
% get the mean bifurcation point
% y1 = nanmean(bifurcation_a) -y0_a;
x1 = 400;
x_apical_a = data.MaxH_1(is_a);
y_apical_a = data.MaxV_1(is_a);
for i_cell = 1: sum(is_a)
    y1 = bifurcation_a(i_cell) + 1/2*y_apical_a(i_cell) - depth_a(i_cell);
    a = x_apical_a(i_cell)/2;
    b = y_apical_a(i_cell)/2;
    t = linspace(-pi,pi,100);
    x = 1 + x1 + a*cos(t);
    y = 1 + y1 + b*sin(t);
    plot(x,y)
    hold on
end
%
axis square
xticks([0, 400])
xticklabels({'b', 'a'})
ylabel('distance from pia (um)')
title('Individual arborization expansion')
box on
grid on
if save_figs
     fig_filename = os.path.join(GC.plot_path, type_experiment , 'Arborization_per_cell_retro.pdf');  
     saveas(fig_all_areas,fig_filename)
     close(fig_all_areas)

end

%% Plot means
%
fig_mean_areas = figure('color', 'w');
%
x0 = 0;
y0 = nanmean(depth_b);
a = nanmean(x_basal_b)/2;
b = nanmean(y_basal_b)/2;
t = linspace(-pi,pi,100);
x = 1 - x0 + a*cos(t);
y = 1 - y0 + b*sin(t);
plot(x,y, 'color', color_b)
hold on
% get the mean bifurcation point
bifurcation_b = T.Bifurcation(is_b);
x1 = 0;
x_apical_b = data.MaxH_1(is_b);
y_apical_b = data.MaxV_1(is_b);
y1 = mean(bifurcation_b + 1/2*y_apical_b - depth_b);

a = nanmean(x_apical_b)/2;
b = nanmean(y_apical_b)/2;
t = linspace(-pi,pi,100);
x = 1 + x1 + a*cos(t);
y = 1 + y1 + b*sin(t);
plot(x,y,'color', color_b)
plot(x0,-y0, 'bx')
plot(x1,y1, 'bx')
% hold on

% plot a
bifurcation_a = T.Bifurcation(is_a);
x0 = 400;
y0 = nanmean(depth_a);
a = nanmean(x_basal_a)/2;
b = nanmean(y_basal_a)/2;
t = linspace(-pi,pi,100);
x = 1 + x0 + a*cos(t);
y = 1 - y0 + b*sin(t);
plot(x,y, 'color', color_a)
hold on

% get the mean bifurcation point
x1 = 400;
x_apical_a = data.MaxH_1(is_a);
y_apical_a = data.MaxV_1(is_a);
y1 = mean(bifurcation_a + 1/2*y_apical_a - depth_a);

a = nanmean(x_apical_a)/2;
b = nanmean(y_apical_a)/2;
t = linspace(-pi,pi,100);
x = 1 + x1 + a*cos(t);
y = 1 + y1 + b*sin(t);
plot(x,y,'color', color_a)
plot(x0,-y0, 'bx')
plot(x1,y1, 'bx')
hold off
%
axis square
xticks([0, 400])
xticklabels({'b', 'a'})
ylabel('distance from soma (um)')
title('Mean arborization expansion')
box on
grid on

%
if save_figs
     fig_filename = os.path.join(GC.plot_path,type_experiment, 'Arborization_means_retro.pdf');  
     saveas(fig_mean_areas,fig_filename)
     close(fig_mean_areas)

end

%% TEST complexity in terms of dim red
fieldnames(T_use) 
b = T_use(is_b, {'MaxV_1', 'MaxH_1', 'MaxOrder', 'Den', 'Oblique' });
b = table2array(b);

a = T_use(is_a, {'MaxV_1', 'MaxH_1', 'MaxOrder', 'Den', 'Oblique' });
a = table2array(a);

% [c_b,s_b,c,d_b,e] = pca(b','NumComponents', 1, 'VariableWeights', 'variance', 'Centered', true);
% [c_a,s_a,~,d_a,~] = pca(a','NumComponents', 1, 'VariableWeights', 'variance', 'Centered', true);

[rows, ~]= find(isnan(b));
b(unique(rows),:)=[];

[rows, ~]= find(isnan(a));
a(unique(rows),:)=[];

%% Norm
% If X is a vector, this is equal to the Euclidean distance
% In particular, the Euclidean distance of a vector from the origin is a norm, 
% called the Euclidean norm, or 2-norm, which may also be defined as the 
% square root of the inner product of a vector with itself.
% 
% Get the values
%b
norm_b =zeros(length(b),1);
for ic = 1:length(b)
norm_b(ic) = norm(b(ic,:));
end
% a
norm_a =zeros(length(a),1);
for ic = 1:length(a)
norm_a(ic) = norm(a(ic,:));
end

[~, p] = ttest2(norm_b, norm_a);

% plot
figure_complexity = figure('color', 'w');

normalized_a = norm_a/ mean(norm_a);
normalized_b = norm_b/ mean(norm_a);
mean_norm_b = mean(normalized_b); sem_norm_b = sem(normalized_b);
mean_norm_a = mean(normalized_a); sem_norm_a = sem(normalized_a);
h = errorbar([mean_norm_a, mean_norm_b], [sem_norm_a, sem_norm_b], 'o', 'LineStyle', 'none', 'Color', 'k');
xlim([0,3])
ylim([0.5 2])
xticks([1 2])
xticklabels(neurons_to_take)
text(1.5, 1.7, ['p=', num2str(p)])
title('Complexity of Arborization index')
ylabel('Normalized complexity (a.i.)')
box off
if save_figs
     fig_filename = os.path.join(GC.plot_path,type_experiment, 'Complexity_a_vs_b_retro.pdf'); 
     saveas(figure_complexity,fig_filename)
     close(figure_complexity)
end




%% Polar plot
data_angle = 90 - data.Polarity;
data_rad_a = deg2rad(data_angle(is_a));
data_rad_b = deg2rad(data_angle(is_b));
% [his, aa] = hist(data_angle);
% fig 1
fig_polar_b = figure('color', 'w') ;
polarhistogram(data_rad_b,'Normalization', 'probability', 'FaceColor', color_b) %  'NumBins', 10
title('Polar plot')
% fig 2
fig_polar_a = figure('color', 'w') ;
polarhistogram(data_rad_a,'Normalization', 'probability','FaceColor',color_a)%;'NumBins', 4
title('Polar plot')

%
if save_figs
     % fig 2
     fig_filename = os.path.join(GC.plot_path,type_experiment, 'Polar_plot_a_retro.pdf');  
     saveas(fig_polar_a,fig_filename)
     close(fig_polar_a)
     % fig 1
     fig_filename = os.path.join(GC.plot_path, type_experiment,'Polar_plot_b_retro.pdf');  
     saveas(fig_polar_b,fig_filename)
     close(fig_polar_b)
end


%% check stats between 'a' and 'b'

% all_features = fieldnames(T);
% all_features(ismember(all_features, {'Experiment','Label','Properties', 'Row', 'Variables'})) = [];
% 
% P_array = cell(0,0);
% for i_f = 1:length(all_features)
%     this_f = all_features{i_f};
%     data_a = T.(this_f)(is_a);
%     data_b = T.(this_f)(is_b);
% 
%     [~, p] = ttest2(data_a, data_b);
%     mean_a = nanmean(data_a);
%     mean_b = nanmean(data_b);
% 
%     P_array(i_f, 1) = {this_f};
%     P_array(i_f, 2) = {p};
%     P_array(i_f, 3) = {[mean_a, mean_b]};
% end
% % covert to table
% variable_names = {'feature', 'p_val', 'a&b'};
% 
% P_table = array2table(P_array, 'VariableNames', variable_names);

%% Separate data into experiments
% 
% categorical(T.Experiment);
% categorical(T.Label);
% % select now only cells b 
% experiments = unique(T.Experiment);
% % CFA_d1_b = T(ismember(T.Experiment, experiments(1)) & ismember(T.Label, 'b'),: );
% % CFA_d7_b = T(ismember(T.Experiment, experiments(2)) & ismember(T.Label, 'b'),: );
% % CFA_d7NS = T(ismember(T.Experiment, experiments(3)) & ismember(T.Label, 'b'),: );
% % Sal_d1_b = T(ismember(T.Experiment, experiments(4)) & ismember(T.Label, 'b'),: );
% % Sal_d7_b = T(ismember(T.Experiment, experiments(5)) & ismember(T.Label, 'b'),: );
% % Sal_d7NS = T(ismember(T.Experiment, experiments(6)) & ismember(T.Label, 'b'),: );
% % 
% % 
% % [~, p]= ttest2(CFA_d7_b.Angle, Sal_d7_b.Angle)
% % [~, p]= ttest2(CFA_d1_b.APThreshold, Sal_d1_b.APThreshold)

%% Plot all data 'a' vs 'b' for all features
% figure_across_tp = figure('color', 'w', 'pos', [100,631,2083,420]);
% F_mean = fieldnames(GS_all_mean);
% F_mean(ismember(F_mean, {'Experiment', 'Label', 'GroupCount', 'Properties', 'Row', 'Properties', 'Variables'})) = [];
% F_std = fieldnames(GS_all_std);
% F_std(ismember(F_std, {'Experiment', 'Label', 'GroupCount', 'Properties', 'Row', 'Properties', 'Variables'})) = [];
% 
% for iif = 1:length(experiments)
%     ax = subplot(floor(length(experiments)/2),3,iif);
%     this_ex = experiments(iif);
%     data_mean = table2array(GS_all_mean(ismember(GS_all_mean.Experiment, this_ex),F_mean));
%     data_sem = table2array(GS_all_std(ismember(GS_all_std.Experiment, this_ex),F_std));
%     n_data = (GS_all_std.GroupCount(ismember(GS_all_std.Experiment, this_ex)));
%     data_sem = (data_sem./sqrt(n_data));
%     errorbar(data_mean', data_sem', 'o','LineStyle', 'none')
%     title (this_ex)
%     xticks([1:length(all_features)])
%     xticklabels(F_mean)
%     xtickangle(45)
% 
% 
% end
% legend({'a', 'b'})
% %
% if save_figs
%      fig_filename = os.path.join(GC.plot_path, type_experiment,'a_vs_b_across_tp_retro.pdf'); 
%      set(figure_across_tp,'PaperSize',[45 15])
%      saveas(figure_across_tp,fig_filename)
%      close(figure_across_tp)
% %      export_fig(fig_filename,figure_across_tp)
% %      print(figure_across_tp,fig_filename,'-dpdf','-r0')
% end
% 
%% Plot data for all experiments, for now only 'b'
% neurons_to_take = {'a';'b'};
% figure_x_exps = figure('color', 'w', 'pos', [100, 200, 2500, 2700]);
% % n_cond = length(experiments)/2; % comparing CFA and sal
% for iif = 1:length(all_features)
%     ax = subplot(5,5,iif);
%     this_f = all_features{iif};
%     this_f_str = ['mean_', this_f];
%     idx_exp = ismember(fieldnames(GS_all_mean), this_f_str);
%     ac = 0;
%     MEAN = [];
%     SEM = [];
%     P = [];
%     for iitp = 1:length(init_tps)
% %         ax = subplot(2,3,iif);
%         this_exp = init_tps(iitp);
%         idx= endsWith(GS_all_mean.Experiment, this_exp) & ismember(GS_all_mean.Label,neurons_to_take);
%         this_data = GS_all_mean(idx,idx_exp);
%         this_labels = GS_all_mean.Experiment(idx);
% 
%         data_mean = table2array(this_data);
%         data_sem = table2array(GS_all_std(idx,idx_exp));
%         exp_1_T = [init_conds{1}, ' ',init_tps{iitp}];
%         data_to_test1 = T(ismember(T.Experiment, exp_1_T) & ismember(T.Label, neurons_to_take(1)), this_f);
%         data_to_test2 = T(ismember(T.Experiment, exp_1_T) & ismember(T.Label, neurons_to_take(2)), this_f);
% 
%         data_to_test1 = table2array(data_to_test1);
%         data_to_test2 = table2array(data_to_test2);
% 
%         n_data = [sum(~isnan(data_to_test1));sum(~isnan(data_to_test2))];
% %         n_data = GS_all_std.GroupCount(idx);
%         data_sem = data_sem./sqrt(n_data);
% 
% %         exp_2_T = [init_conds{2}, ' ',init_tps{iitp}];
%         [~, p] = ttest2(data_to_test1, data_to_test2);
%         if height(data_to_test1) < 3 || height(data_to_test2) < 3, p = 1; end
%         MEAN = [MEAN,data_mean];
%         SEM = [SEM,data_sem];
%         P = [P ,p];
%     end
% 
%     h = errorbar( MEAN', SEM', 'o','LineStyle', 'none', 'Color',color_a);
%     set(h(2), 'Color',  color_b)
%     xlim([0,4])
%     xticks([1:3])
%     xticklabels(init_tps)
%     xtickangle(45)
%     legend(neurons_to_take)
%     text([1], max(MEAN(:)), num2str(P))
%     title(this_f)
% 
% end
% %
% if save_figs
%      fig_filename = os.path.join(GC.plot_path,type_experiment, 'a_vs_b_across_tp_retro.pdf'); 
%      set(figure_x_exps,'PaperSize',[60 45])
%      saveas(figure_x_exps,fig_filename)
%      close(figure_x_exps)
% end

% %% Combining different timepoints
% 
% neurons_to_take = {'a';'b'};
% figure_x_fea = figure('color', 'w', 'pos', [100, 200, 2500, 2700]);
% % n_cond = length(experiments)/2; % comparing CFA and sal
% 
% data_a = T_use(is_a,:);
% data_b = T_use(is_b,:);
% 
% 
% 
% 
% for iif = 1:length(all_features)
%     ax = subplot(5,5,iif);
%     this_f = all_features{iif};
% %     this_f_str = ['mean_', this_f];
%     this_data_a = data_a.(this_f);
%     this_data_b = data_b.(this_f);
% 
%     mean_a = nanmean(this_data_a);
%     sem_a = sem(this_data_a);
% 
%     mean_b =nanmean(this_data_b);
%     sem_b = sem(this_data_b);
% 
% 
%     [~, p] = ttest2(this_data_a, this_data_b);
%     if length(this_data_a) < 3 || length(this_data_b) < 3, p = []; end
% 
%     h = errorbar( [mean_a mean_b], [sem_a sem_b], 'o','LineStyle', 'none', 'Color', 'k');
% %     set(h(2), 'Color', [0 0 1])
%     xlim([0,3])
%     xticks([1 2])
%     xticklabels(neurons_to_take)
%     xtickangle(0)
% %     legend(neurons_to_take)
%     text([1], max(mean_a(:)), num2str(p))
%     title(this_f)
%     box off
%     axis square
% end
% 
% if save_figs
%      fig_filename = os.path.join(GC.plot_path, type_experiment,'a_vs_b_all_in_one_x_f_retro.pdf'); 
%      set(figure_x_fea,'PaperSize',[60 45])
%      saveas(figure_x_fea,fig_filename)
%      close(figure_x_fea)
% end







