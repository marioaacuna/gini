function [T_train, T_pred] = load_input_table(filename_table )
    global GC
    table_filename = fullfile(GC.raw_data_folder, 'in',filename_table);
    opts = detectImportOptions(table_filename);
    parameters = opts.VariableNames;
    % model to use
    
    GC.variables_to_discard = {'Experiment', 'ID', 'Slice', 'Date', 'SpikeCount', 'IInj'};
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
    vars_to_eval = [GC.variables_to_evaluate,'ADP', 'Label'];
    % delete burst
    
    %vars_to_eval = [vars_to_eval, 'ADP'];
    T_train = T_new(:,vars_to_eval);
    
    %% Read input table
    
    table_filename = fullfile(GC.raw_data_folder,'in', "input.xlsx");
    GC.variables_to_discard = {'Label','Experiment', 'ID', 'Slice', 'Date', 'SpikeCount', 'IInj'};
    
    % Variables = parameters(~ismember(parameters, variables_to_discard)); 
    % opts.SelectedVariableNames = Variables;
    %T = readtable( table_filename,opts);
    
    % Read table
    T = readtable(table_filename);
    opts.SelectedVariableNames = 'Label';
    
    % get neurons only in L5
    % threshold = GC.threshold_depth_L5; % Thomas' paper
    threshold = 350;
    T(T.Depth < threshold, :) =[];
    
    % make sure some nan values are set to 0 (not present)
    T.Burst(isnan(T.Burst)) = 0;
    T.ICAmp(isnan(T.ICAmp))= 0;
    T_pred = T;
    % %% Run tree
    % tree_1 = fitrtree(T_train(:,ismember(T_train.Properties.VariableNames, vars_to_eval)), response,'PredictorSelection','curvature','Surrogate','on');
    % % Plot figure importance
end    
