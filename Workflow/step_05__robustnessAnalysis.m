clc
clear
global GC
%% read retro table
[T_train,T_pred] = load_input_table('Retro_ACC_PAG.xlsx');

mdl_filename = fullfile(GC.raw_data_folder, 'out','mdl_LR_Signif_preds.mat');
%% model
% load model if exists
mdl = load_variable(mdl_filename, 'mdl');

%% Load Tree estimatiors
pred_filename = fullfile(GC.raw_data_folder, 'out', 'TREE_predictors.mat');
preds = load_variable(pred_filename, 'TREE_predictors');
S = preds.S;
norm_estimates = preds.norm_estimates(S);
pred_names = preds.pred_names(S);



data_to_use_for_index = T_train(:, pred_names);
d = table2array(data_to_use_for_index);
it = 1;
ACC = [];
C  = [];
num_iterations = 1000;
while it <= num_iterations
    % Compute data augmentation
    % Assuming 'myData' is your dataset matrix
    noise_factor_fraction = 0.9;  % For example, you want to use 10% of the CV as noise
    [cvs, noise_factors] = calculateCVandNoiseFactors(d, noise_factor_fraction);

    % Do data augmentation
    augmented_data = augmentData(d, noise_factors);

    %% Retrain the classifier
    % concatenate the original data with the augmented data
    d_c = [d; augmented_data];

    % normalize the values
    d_z= zscore(d_c);

    % Do PCA
    pca_to_retrain = do_pca_gini(d_z, norm_estimates);
    labels = [T_train.Label;T_train.Label];



    % % Comment:
    % % Sadly on Matlab we cannot retrain the same model, what we are going to do is to obtain the parameters from the model and retrain it from scratch
    % % Get the parameters from the model
    % 
    % % Extract hyperparameters from the old model
    % %
    % 
    % 
    % % Assuming `mdl` is your original model structure
    % originalModel = mdl.ClassificationKernel;
    % 
    % % Extract parameters
    % boxConstraint = originalModel.BoxConstraint;
    % classNames = originalModel.ClassNames;
    % prior = originalModel.Prior;
    % cost = originalModel.Cost;
    % lambda = originalModel.Lambda;
    % regularization = originalModel.Regularization;
    % kernelScale = originalModel.KernelScale;
    % learner = originalModel.Learner;  % Though this might always be 'logistic' for logistic regression
    % 
    % % Prepare your combined data (original + augmented)
    % % Assuming 'X' is your predictors and 'Y' is your response
    % combinedX = pca_to_retrain;
    % combinedY =labels;
    % % Assuming `combinedX` and `combinedY` are your predictors and response data
    % nFolds = 5; % Number of folds for k-fold cross-validation
    % cv = cvpartition(combinedY, 'KFold', nFolds);
    % 
    % % Preallocate arrays to store results for each fold
    % accuracy = zeros(nFolds, 1);
    % confMatrices = cell(nFolds, 1);
    % 
    % % Array to store all predictions and actual values
    % allPredictions = [];
    % allTrueLabels = [];
    % allIndices = [];  % Store indices of validation samples
    % 
    % for i = 1:nFolds
    %     % Training data for the current fold
    %     trainX = combinedX(cv.training(i), :);
    %     trainY = combinedY(cv.training(i), :);
    % 
    %     % Validation data for the current fold
    %     valX = combinedX(cv.test(i), :);
    %     valY = combinedY(cv.test(i), :);
    % 
    %     % Train the model - if needed, uncomment and use a training command
    %     % model = trainModelFunction(trainX, trainY); % Replace with actual training function
    % 
    %     % Validate the model
    %     predictions = mdl.predictFcn(valX);
    % 
    %     % Store the predictions and actual values
    %     allPredictions = [allPredictions; predictions];
    %     allTrueLabels = [allTrueLabels; valY];
    %     allIndices = [allIndices; find(cv.test(i))];  % Store the indices of the validation set
    % 
    %     % Calculate accuracy
    %     accuracy(i) = sum(predictions == valY) / length(valY);
    % 
    %     % Generate and store the confusion matrix for the current fold
    %     confMatrices{i} = confusionmat(valY, predictions);
    % end
    % 
    % % Plotting correct and incorrect predictions
    % % Find correct predictions
    % correctPredictions = (allPredictions == allTrueLabels);
    % 

    %% Plot results
    % figure
    % gscatter(pca_to_retrain(allIndices,1), pca_to_retrain(allIndices,2), allPredictions, [], 'o', 12)
    % hold on
    % gscatter(pca_to_retrain(allIndices,1), pca_to_retrain(allIndices,2), allTrueLabels, [], 'x')
    % title(['ACC : ', num2str(mean(accuracy))])


    %% Alternatively run it at once
    new_predictions = mdl.predictFcn(pca_to_retrain);
    % figure
    % gscatter(pca_to_retrain(:,1), pca_to_retrain(:,2), new_predictions, [], 'o', 12)
    % hold on
    % gscatter(pca_to_retrain(:,1), pca_to_retrain(:,2), labels, [], 'x')
    acc = sum(new_predictions == labels) / length(labels);
    % title(['ACC : ', num2str(acc)], 'Color', 'w')
    c = confusionmat(labels, new_predictions);
    c = c./sum(c,2);
    C = cat(3,C,c);

    % do_figures_black(gca, 0)
    % set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'GridColor', 'w'); % Set the current axes properties
    % set(gcf, 'Color', 'w'); % Set the current figure background
    % % set(findall(gca, 'Type', 'Line'), 'Color', 'w'); % Set line plot colors to white
    % box off
    % % Set the ticks direction to outside
    % set(gca, 'TickDir', 'out', 'TickLength', [0.02, 0.02]); % Adjust the tick length as desired


    %%
    % Display the average accuracy across all folds
    % averageAccuracy = mean(accuracy);
    % disp(['Average Accuracy: ', num2str(averageAccuracy)]);
    ACC = [ACC; acc];
    it = it + 1;
end
% end loop here
keyboard
%%
% Number of classes
numClasses = 2;

% Initialize matrix to hold summed normalized confusion matrices
summedConfMatrix = zeros(numClasses, numClasses);

% Loop over each fold's confusion matrix
for i = 1:length(confMatrices)
    % Normalize confusion matrix by the number of observations in each class
    normConfMatrix = confMatrices{i} ./ sum(confMatrices{i}, 2);

    % Sum the normalized confusion matrices
    summedConfMatrix = summedConfMatrix + normConfMatrix;
end

% Calculate the mean normalized confusion matrix
meanConfMatrix = summedConfMatrix / nFolds;

% Plotting the mean normalized confusion matrix as a heatmap
% figure;
% heatmap(meanConfMatrix, 'Colormap');
% title('Mean Normalized Confusion Matrix');
% xlabel('Predicted Class');
% ylabel('True Class');
%
% % Optional: Display the values on the heatmap
% textStrings = num2str(meanConfMatrix(:), '%0.2f');       % Create strings from the matrix values
% textStrings = strtrim(cellstr(textStrings));             % Remove any space padding
% [x, y] = meshgrid(1:numClasses);                         % Create x and y coordinates for the strings

%{
 hStrings = text(x(:), y(:), textStrings(:), ...          % Plot the strings
                'HorizontalAlignment', 'center');
colorbar; 
%}

%%

keyboard
%%


%% write table
T_to_write = [T(~is_nan_idx,:), table(yfit, 'VariableNames',{'Mdl_predictors'})];
new_table_filename = fullfile(GC.raw_data_folder,'out',"input_with_predicted_lables_LR_Signif_preds_grouped.xlsx");
if exist(new_table_filename, 'file')
    disp('It will re-write table')
    delete(new_table_filename)
end
writetable(T_to_write, new_table_filename)

%% Reorganize data
keyboard % TODO
step_03__fig2_reorganizedata2excel_from_Mdl(T_to_write)

