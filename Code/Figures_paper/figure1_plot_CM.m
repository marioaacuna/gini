% figure confusion matrix predictions figure 1
% this script will test the predictions, but the validation CM is the
% same. So for simplicity we do it so.
%% Load data
in_datafile = fullfile("Data/in/Retro_ACC_PAG.xlsx");
T = readtable(in_datafile);

%% load model
mdl_filename = fullfile("Data/out/mdl_LR_Signif_preds.mat");
load(mdl_filename)

%% Load predictorys
pred_filename = fullfile(GC.raw_data_folder, 'out', 'TREE_predictors.mat');
preds = load_variable(pred_filename, 'TREE_predictors');
S = preds.S;
norm_estimates = preds.norm_estimates(S);
pred_names = preds.pred_names(S);

%% run pca
data_to_use_for_index = T(:, pred_names);
d = table2array(data_to_use_for_index);
d_z= zscore(d);
pca_d = do_pca_gini(d_z, norm_estimates);

%% run pred

[yfit_nb,~] = mdl.predictFcn(pca_d);

%% Plot CM
figure(1), confusionchart(T.Label,yfit_nb, "Normalization","row-normalized")

%%
figure(2)
cm = confusionmat(T.Label,yfit_nb);
cm2 = cm./sum(cm,2) *100;
heatmap(cm2)