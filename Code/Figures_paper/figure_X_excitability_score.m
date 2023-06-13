% compute exitability score
%% Preambple
% this script will attemp to produce similar results as in Falkowska et al
% 2023, where given values of APthreshold, InputR (and maybe others) we
% could get an excitability score to then compare accross groups

%% Load retro and input table
label_to_eval = 1; % or 0


excitability_vars = {'InputR', 'APThreshold', 'SpikeCount', 'SAG', 'ADP', 'Tau'};

table_filename = fullfile(GC.raw_data_folder, 'in','Retro_ACC_PAG.xlsx');

% Read table
T = readtable(table_filename);
opts.SelectedVariableNames = 'Label';

% get neurons only in L5
threshold = GC.threshold_depth_L5; % Thomas' paper
T(T.Depth < threshold, :) =[];
T_new = T;
% get the excitability variables;
vars_to_take = ['Label', excitability_vars];
T_retro = T_new(:,ismember(T_new.Properties.VariableNames, vars_to_take));
T_retro.Experiment = repmat({'Naive'},height(T_retro),1 );
T_retro = T_retro(ismember(T_retro.Label, label_to_eval),:);
%% Load input
table_filename = fullfile(GC.raw_data_folder,'out', "input_with_predicted_lables_LR_Signif_preds_grouped.xlsx");

% Read table
T = readtable(table_filename);

threshold = 350;
T(T.Depth < threshold, :) =[];

T_blind = T;

vars_to_take = ['Mdl_predictors', excitability_vars, 'Experiment'];
T_blind = T_blind(:,ismember(T_blind.Properties.VariableNames, vars_to_take));
% change name to Label
T_blind.Properties.VariableNames(ismember(T_blind.Properties.VariableNames, 'Mdl_predictors')) = {'Label'};

%% Convert values of APthr, InputR and so on into a score
T_total = [T_retro;T_blind];
isnan_idx = isnan(table2array(T_total(:,'InputR'))) | (isnan(table2array(T_total(:,'APThreshold'))));
% delete the nan entries
T_total(isnan_idx,:) = [];
% z_data_ephys = [zscore(table2array(T_total(~isnan_idx,'InputR'))), -zscore(table2array(T_total(~isnan_idx,'APThreshold')))];% 1: InputR, 2: APThr
if any(isnan(zscore(table2array(T_total(:,'SpikeCount')))))
    % in case of taking spike count with nan values:
    spkdata =table2array(T_total(:,'SpikeCount'));
    zscore_spk = (spkdata - nanmean(spkdata)) / nanstd(spkdata);

else
    zscore_spk = zscore(table2array(T_total(:,'SpikeCount')));
    % z_data_ephys = [zscore(table2array(T_total(:,'InputR'))), -zscore(table2array(T_total(:,'APThreshold'))), zscore(table2array(T_total(:,'SpikeCount')))];% 1: InputR, 2: APThr
end

z_data_ephys = [...
    zscore(table2array(T_total(:,'InputR'))), ...
    -zscore(table2array(T_total(:,'APThreshold'))),...
    -zscore(table2array(T_total(:,'Tau'))),...
    zscore(table2array(T_total(:,'ADP'))),...
    zscore(table2array(T_total(:,'SAG'))),...
    zscore_spk,...

    ];% 1: InputR, 2: APThr




% compute the excitability score based on the average of the ex variables
ex_score = nanmedian(z_data_ephys,2);
%% add it to the table 
T_total.Excitability = ex_score;


%% Evaluate scores
% T_total.Label = categorical(T_total.Label);
% T_total.Experiment = categorical(T_total.Experiment);


experiments = unique(T_total.Experiment);
if any(ismember(experiments, {'','Experiment'}))
    experiments(ismember(experiments, {'','Experiment'})) = [];
end

this_label = label_to_eval;
% Loop through the experiments)
figure
data_all = [];
idx_all = [];
for iex = 1:length(experiments)
    % init experiment
    this_exp = experiments{iex};
    this_data = T_total.Excitability(ismember(T_total.Experiment, this_exp) & T_total.Label == (this_label));
    scatter(repmat(iex, length(this_data),1), this_data)
    hold on
    idx_all = [idx_all;repmat(iex,length(this_data),1)];
    data_all = [data_all;this_data];
    % this_data = T(ismember(T.Experiment, this_exp) & ismember(T.Mdl_predictors, this_label),:);
    % writetable(this_data, filename, "Sheet",this_exp, 'WriteMode', 'overwritesheet')
end

xticklabels(experiments)
hold off
%% Normalize to naive
idx_naive =find(ismember(experiments, 'Naive'));
mean_naive = nanmean(data_all(ismember(idx_all, idx_naive)));
new_data_all = data_all-mean_naive;

%% plot
figure('color', 'w')
% swarmchart(x,y)
swarmchart(idx_all, new_data_all, 'XJitterWidth',0.33)
hold on
xticks([1:7])
plot([1:7], [repmat(0,7,1)])
xticklabels(experiments)
set(gca, 'TickDir','out')
ylabel('Norm Excitability')
title('SC neurons')



%% quick check
idx_sald1 =find(ismember(experiments, 'Saline d1'));
idx_sald7 =find(ismember(experiments, 'Saline d7'));
idx_sald7NS =find(ismember(experiments, 'Saline d7NS'));

idx_cfad1 =find(ismember(experiments, 'CFA d1'));
idx_cfad7 =find(ismember(experiments, 'CFA d7'));
idx_cfad7NS = find(ismember(experiments, 'CFA d7NS'));


ttest2(new_data_all(ismember(idx_all, idx_naive)), new_data_all(ismember(idx_all, idx_sald1)))
ttest2(new_data_all(ismember(idx_all, idx_naive)), new_data_all(ismember(idx_all, idx_sald7)))
ttest2(new_data_all(ismember(idx_all, idx_naive)), new_data_all(ismember(idx_all, idx_cfad7)))
ttest2(new_data_all(ismember(idx_all, idx_naive)), new_data_all(ismember(idx_all, idx_cfad7NS)))
[~, p]=ttest2(new_data_all(ismember(idx_all, idx_cfad7)), new_data_all(ismember(idx_all, idx_cfad7NS)))
ttest2(new_data_all(ismember(idx_all, idx_naive)), new_data_all(ismember(idx_all, idx_cfad1)))
ttest2(new_data_all(ismember(idx_all, idx_cfad7)), new_data_all(ismember(idx_all, idx_cfad1)))
[~, p]=ttest2(new_data_all(ismember(idx_all, idx_cfad7NS)), new_data_all(ismember(idx_all, idx_sald7NS)))

ttest2(new_data_all(ismember(idx_all, idx_sald7)), new_data_all(ismember(idx_all, idx_sald1)))
[~, p]=ttest2(new_data_all(ismember(idx_all, idx_sald7)), new_data_all(ismember(idx_all, idx_sald7NS)))




 [p]=wilcoxon_ranksum(new_data_all(ismember(idx_all, idx_cfad7)), new_data_all(ismember(idx_all, idx_cfad7NS)))
%%
anova1(new_data_all, idx_all),
hold on
xticks([1:7])
plot([1:7], [repmat(0,7,1)], 'k--')
xticklabels(experiments)
set(gca, 'TickDir','out')
ylabel('Norm Excitability')
title('SC neurons')

