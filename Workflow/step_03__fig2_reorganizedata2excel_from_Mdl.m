%% Preamble
% This script will help categorization of dataset regarding the CFA/ SAL
% project and IT and SC neurons, taken the inout excel file and pulling out
% an excel file with the right organizations for the figures. These excel
% files will be used to plot te figures on Prism 
%% INIT
function step_03__fig2_reorganizedata2excel_from_Mdl(T)
disp('Writing table')
% clear
clc
global GC

%% Read input table
% table_filename = fullfile(GC.raw_data_folder,'out',"input_with_predicted_lables_LR_signif_preds.xlsx");
% T = readtable(table_filename);
%% Categorize the different groups
% get the variables
vars_to_evaluate = GC.variables_to_evaluate;
% it might be that not all the variables are present

% loop though the labels
labels = unique(T.Mdl_predictors);
% each sheet will have an experiment, and we will have as many excels as
% labels

for il = 1:length(labels)
    % init label
    this_label = labels(il);
    filename = fullfile(GC.raw_data_folder,'out', ['Re-organization_', num2str(this_label), '.xlsx']);
    if exist(filename, 'file')
        delete(filename)
    end
    % it will get all the data for this label
    lab_idx = T.Mdl_predictors == this_label;

    experiments = unique(T.Experiment);
    if any(ismember(experiments, {'','Experiment'}))
        experiments(ismember(experiments, {'','Experiment'})) = [];
    end
        % Loop through the experiments)
    for iex = 1:length(experiments)
        % init experiment
        this_exp = experiments{iex};
        this_data = T(ismember(T.Experiment, this_exp) & ismember(T.Mdl_predictors, this_label),:);
        writetable(this_data, filename, "Sheet",this_exp, 'WriteMode', 'overwritesheet')
    end
   
    



end


 disp('done reorganization')
