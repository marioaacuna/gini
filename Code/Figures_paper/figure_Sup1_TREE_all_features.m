%% preamble
% Script to plot the horizontal bars for all features
% Figure SUP1
%%
clear
clc
close all

% %% Init variables
% save_fig = 0 ; % Whether or not save the figures as pdf in the 
% save_predictors = 0; % Whether or not save the Index
% n_predictors = 5; % Number of predictors used to generate the Index

%%
FP = figure_properties();
type_experiment = 'retro';
color_a = FP.colors.groups.a;
color_b = FP.colors.groups.b;

global GC
GC = general_configs();
% read table
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
vars_to_eval = [vars_to_eval, 'ADP'];
T_use = T_new(:,vars_to_eval);

%% Run a tree classifier to determine the important variables
tree_1 = fitrtree(T_new(:,ismember(T_new.Properties.VariableNames, vars_to_eval)), response,'PredictorSelection','curvature','Surrogate','on');
% Plot figure importance
imp = predictorImportance(tree_1);


%% plot figure

figure_predictors = figure('Color','w');
[imp_sorted, sorted_idx] = sort(imp, 'ascend');
is_zero = imp_sorted==0;
estimates = imp_sorted;
norm_estimates = estimates / max(estimates);
pred_names = tree_1.PredictorNames(sorted_idx);

barh(norm_estimates);
title('Predictor Importance Estimates');
xlabel('Estimates');
ylabel('Predictors');
h = gca;

% pred_names = pred_names(logical(S));
yticks([1:length(pred_names)])
h.YTickLabel = pred_names;
% h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
box off
