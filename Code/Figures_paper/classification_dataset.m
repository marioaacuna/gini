% evaluate from new blind obtained data
clear, clc

global GC
%% analyse Blind-obtained neurons
% table_filename = os.path.join(GC.raw_data_folder, 'Only_saline_data.xlsx');
table_filename = fullfile(GC.raw_data_folder, "input.xlsx");
GC.variables_to_discard = {'Label', 'Experiment', 'ID', 'Slice', 'Date', 'SpikeCount', 'IInj'};

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
% TODO: complete the burst data

%T_new.Burst(isnan(T_new.Burst)) = 0;
original_labels = T_new.Label;
% actual labels for REtroACCPAG data are [0,1] instead of [a, b].
response = double(ismember(original_labels, 'b'));
%vars_to_eval = GC.variables_to_evaluate;

%% Load predictors
pred_filename = fullfile(GC.raw_data_folder,"out/predictor_weights.mat");
predictors = load_variable(pred_filename, 'predictors');
vars_to_eval = predictors.names;

T_use = T_new(:,vars_to_eval);


%% Get indexes
indexes = calculate_indexes(predictors, T_use);
class_threshold = predictors.threshold;

figure, gscatter(indexes, repmat(1, size(indexes)), response, [], 'o')
hold on
plot([class_threshold,class_threshold   ], [0,2], '--')
hold off


is_1 = double(indexes>class_threshold);

%% write table
new_T = [T, table(is_1, 'VariableNames',{'LabelPrediction'})];;
new_table_filename = fullfile(GC.raw_data_folder,'out',"input_with_predicted_lables.xlsx");
writetable(new_T, new_table_filename)
