%% Preamble
% This script will make a quick characterization between experiments
%% Init
clear, clc
global GC
%%
r = 1; % type of neuron (1 , 0)
table_filename = fullfile(GC.raw_data_folder,'out', "input_with_predicted_lables.xlsx");
% Read table
addpath(fullfile(pwd, '\Code\Utilities\Statistics'))
T = readtable(table_filename);

% get neurons only in L5
threshold = GC.threshold_depth_L5; % Thomas' paper
T(T.Depth < threshold, :) =[];
T_new = T;
% make sure some nan values are set to 0 (not present)
% clT_new.ICAmp(isnan(T_new.ICAmp))= 0;
% TODO: complete the burst data

%T_new.Burst(isnan(T_new.Burst)) = 0;
% response = T_new.Label;
% if any(ismember(response, 'a'))
%     response = double(ismember(response, 'b'));
% end

response = double(T_new.LabelPrediction);
% vars_to_eval = GC.variables_to_evaluate;
vars_to_eval  = GC.variables_to_evaluate;
experiments = {'CFA d7',...
            'CFA d7NS'};

%% Plot
f1 = figure('Position',[0 0 2500 2000]);
ncols = 14;
nrows = 2;
for i_var = 1:length(vars_to_eval)
    try
        this_var = vars_to_eval{i_var};
        ex1 = table2array(T(ismember(T.Experiment,experiments{1}) & response ==r, this_var));
        ex2 = table2array(T(ismember(T.Experiment,experiments{2}) & response ==r, this_var));
        % do stats
        [~, p]=ttest2(ex2, ex1);
        subplot(nrows, ncols, i_var)
        errorbar([1,2],[nanmean(ex1), nanmean(ex2)], [sem(ex1), sem(ex2)], 'LineStyle','none', 'Marker','o', 'MarkerFaceColor','auto')
        xlim([0,3])
        xticks([1,2])
        xticklabels(experiments)
        text(1.5, mean([nanmean(ex1), nanmean(ex2)]), num2str(p))
        title(this_var)
    catch ME
        disp(ME.message)
        continue
    end
end


%% plot sal to cfa d1
experiments = {'CFA d1',...
            'Saline d1' };


f1 = figure('Position',[0 0 2500 2000]);
ncols = 14;
nrows = 2;
for i_var = 1:length(vars_to_eval)
    try
        this_var = vars_to_eval{i_var};
        ex1 = table2array(T(ismember(T.Experiment,experiments{1}) & response ==r, this_var));
        ex2 = table2array(T(ismember(T.Experiment,experiments{2}) & response ==r, this_var));
        % do stats
        [~, p]=ttest2(ex2, ex1);
        subplot(nrows, ncols, i_var)
        errorbar([1,2],[nanmean(ex1), nanmean(ex2)], [sem(ex1), sem(ex2)], 'LineStyle','none', 'Marker','o', 'MarkerFaceColor','auto')
        xlim([0,3])
        xticks([1,2])
        xticklabels(experiments)
        text(1.5, mean([nanmean(ex1), nanmean(ex2)]), num2str(p))
        title(this_var)
    catch ME
        disp(ME.message)
        continue
    end
end


%% plot sal to cfa d7
experiments = {'CFA d7',...
            'Saline d7' };


f1 = figure('Position',[0 0 2500 2000]);
ncols = 14;
nrows = 2;
for i_var = 1:length(vars_to_eval)
    try
        this_var = vars_to_eval{i_var};
        ex1 = table2array(T(ismember(T.Experiment,experiments{1}) & response ==r, this_var));
        ex2 = table2array(T(ismember(T.Experiment,experiments{2}) & response ==r, this_var));
        % do stats
        [~, p]=ttest2(ex2, ex1);
        subplot(nrows, ncols, i_var)
        errorbar([1,2],[nanmean(ex1), nanmean(ex2)], [sem(ex1), sem(ex2)], 'LineStyle','none', 'Marker','o', 'MarkerFaceColor','auto')
        xlim([0,3])
        xticks([1,2])
        xticklabels(experiments)
        text(1.5, mean([nanmean(ex1), nanmean(ex2)]), num2str(p))
        title(this_var)
    catch ME
        disp(ME.message)
        continue
    end
end

%%
experiments = {'Saline d7NS',...
            'CFA d7NS'};
% Plot
f1 = figure('Position',[0 0 2500 2000]);
ncols = 14;
nrows = 2;
for i_var = 1:length(vars_to_eval)
    try
        this_var = vars_to_eval{i_var};
        ex1 = table2array(T(ismember(T.Experiment,experiments{1}) & response ==r, this_var));
        ex2 = table2array(T(ismember(T.Experiment,experiments{2}) & response ==r, this_var));
        % do stats
        [~, p]=ttest2(ex2, ex1);
        subplot(nrows, ncols, i_var)
        errorbar([1,2],[nanmean(ex1), nanmean(ex2)], [sem(ex1), sem(ex2)], 'LineStyle','none', 'Marker','o', 'MarkerFaceColor','auto')
        xlim([0,3])
        xticks([1,2])
        xticklabels(experiments)
        text(1.5, mean([nanmean(ex1), nanmean(ex2)]), num2str(p))
        title(this_var)
    catch ME
        disp(ME.message)
        continue
    end
end


