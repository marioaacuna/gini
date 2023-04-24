
%% Preamble
% General script to run different plots. Mean area plot, Polar plot, and
% several comparisons

clear
clc
close all
warning off

global GC
%% Inputs
take_only_saline = 0;
save_figs = 1;

%% Get the data

input_file = os.path.join(GC.raw_data_folder, 'input.xlsx');
% input_file =  'M:\Mario\Gini\dataset.xlsx'; % from 2019 dataset
opts = detectImportOptions(input_file);
init_conds = GC.init_conds.CFA;
init_tps = GC.init_tps.CFA;

parameters = opts.VariableNames;
variables_to_discard =  GC.variables_to_discard; % , 'Burst', 'ICAmp'
Variables = parameters(~ismember(parameters, variables_to_discard)); 
% Variables = parameters(ismember(parameters, {'SAG', 'Diameter'}));

opts.SelectedVariableNames = Variables;
T = readtable( input_file,opts);
% opts.SelectedVariableNames = 'Label';
% L = readtable(input_file,opts);
threshold = GC.throshold_depth_L5; % Thomas' paper
T(T.Depth < threshold, :) =[];
%% Select variables to plot
% fieldnames(T)
if take_only_saline
    init_conds = init_conds(1);
end
in_idx = startsWith(T.Experiment, init_conds); 
T = T(in_idx,:);


%%
is_b = ismember(T.Label, 'b') & T.Depth > threshold;
is_a = ~ismember(T.Label, 'b') &  T.Depth > threshold;

GS_all_mean = groupsummary(T, {'Experiment', 'Label'}, 'mean', 'IncludeMissingGroups', false);
GS_all_std = groupsummary(T, {'Experiment', 'Label'}, 'std', 'IncludeMissingGroups', false);


all_features = fieldnames(T);
all_features(ismember(all_features, {'Experiment','Label','Properties', 'Row', 'Variables'})) = [];

P_array = cell(0,0);
for i_f = 1:length(all_features)
    this_f = all_features{i_f};
    data_a = T.(this_f)(is_a);
    data_b = T.(this_f)(is_b);
    
    [~, p] = ttest2(data_a, data_b);
    mean_a = nanmean(data_a);
    mean_b = nanmean(data_b);
    
    P_array(i_f, 1) = {this_f};
    P_array(i_f, 2) = {p};
    P_array(i_f, 3) = {[mean_a, mean_b]};
end
% covert to table
variable_names = {'feature', 'p_val', 'a&b'};

P_table = array2table(P_array, 'VariableNames', variable_names);

%% Separate data into experiments

categorical(T.Experiment);
categorical(T.Label);
% select now only cells b 
experiments = unique(T.Experiment);


%% Plot data for all experiments, for now only 'b'
neurons_to_take = {'b'};
figure_x_exps = figure('color', 'w', 'pos', [100, 200, 2500, 2700]);
% n_cond = length(experiments)/2; % comparing CFA and sal
for iif = 1:length(all_features)
    ax = subplot(5,5,iif);
    this_f = all_features{iif};
    this_f_str = ['mean_', this_f];
    idx_exp = ismember(fieldnames(GS_all_mean), this_f_str);
    ac = 0;
    MEAN = [];
    SEM = [];
    P = [];
    for iitp = 1:length(init_tps)
%         ax = subplot(2,3,iif);
        this_exp = init_tps(iitp);
        idx = endsWith(GS_all_mean.Experiment, this_exp) & ismember(GS_all_mean.Label,neurons_to_take);
        this_data = GS_all_mean(idx,idx_exp);
        this_labels = GS_all_mean.Experiment(idx);
        
        data_mean = table2array(this_data);
        data_sem = table2array(GS_all_std(idx,idx_exp));
        n_data = GS_all_std.GroupCount(idx);
        data_sem = data_sem./sqrt(n_data);
        exp_1_T = [init_conds{1}, ' ',init_tps{iitp}];
        exp_2_T = [init_conds{2}, ' ',init_tps{iitp}];
        data_to_test1 = T(ismember(T.Experiment, exp_1_T) & ismember(T.Label, neurons_to_take), this_f); 
%         data_to_test2 = T(ismember(T.Experiment, exp_2_T) & ismember(T.Label, neurons_to_take), this_f); 
%         [~, p] = ttest2(table2array(data_to_test1), table2array(data_to_test2));
        
        MEAN = [MEAN,data_mean];
        SEM = [SEM,data_sem];
%         P = [P ,p];
    end
    
    errorbar( MEAN', SEM', 'o','LineStyle', 'none')
    xlim([0,4])
    xticks([1:3])
    xticklabels(init_tps)
    xtickangle(45)
    legend(init_conds)
    text([1], max(MEAN(:)), num2str(P))
    title(this_f)

end


%%
if save_figs
     fig_filename = os.path.join(GC.plot_path, 'b_across_tp_SAL_CFA.pdf'); 
     set(figure_x_exps,'PaperSize',[60 45])
     saveas(figure_x_exps,fig_filename)
     close(figure_x_exps)
end


