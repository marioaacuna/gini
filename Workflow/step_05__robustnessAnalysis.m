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



% Comment:
% Sadly on Matlab we cannot retrain the same model, what we are going to do is to obtain the parameters from the model and retrain it from scratch
% Get the parameters from the model

% Extract hyperparameters from the old model
%


% Assuming `mdl` is your original model structure
originalModel = mdl.ClassificationKernel;

% Extract parameters
boxConstraint = originalModel.BoxConstraint;
classNames = originalModel.ClassNames;
prior = originalModel.Prior;
cost = originalModel.Cost;
lambda = originalModel.Lambda;
regularization = originalModel.Regularization;
kernelScale = originalModel.KernelScale;
learner = originalModel.Learner;  % Though this might always be 'logistic' for logistic regression

% Prepare your combined data (original + augmented)
% Assuming 'X' is your predictors and 'Y' is your response
combinedX = pca_to_retrain;
combinedY =labels;

% do 5-fold cross validation
% Assuming `combinedX` and `combinedY` are your predictors and response data
nFolds = 5; % Number of folds for k-fold cross-validation
cv = cvpartition(combinedY, 'KFold', nFolds);

% Preallocate arrays to store results for each fold
accuracy = zeros(nFolds, 1);
confMatrices = cell(nFolds, 1);

for i = 1:nFolds
    % Training data for the current fold
    trainX = combinedX(cv.training(i), :);
    trainY = combinedY(cv.training(i), :);
    
    % Validation data for the current fold
    valX = combinedX(cv.test(i), :);
    valY = combinedY(cv.test(i), :);
    
    
    
%{
 % Train the model
    model = fitclinear(trainX, trainY, ...
            'ClassNames', classNames, ...
            'Prior', prior, ...
            'Cost', cost, ...
            'Lambda', lambda, ...
            'Regularization', 'ridge', ...
            'Learner', learner, ...
            'ObservationsIn', 'rows', ...
            'ScoreTransform', originalModel.ScoreTransform);
     
%}

    % Validate the model
    %predictions = predict(mdl, valX);
    predictions = mdl.predictFcn(valX);
    
    % Calculate accuracy
    accuracy(i) = sum(predictions == valY) / length(valY);
    
    % Generate and store the confusion matrix for the current fold
    confMatrices{i} = confusionmat(valY, predictions);
end

% Display the average accuracy across all folds
averageAccuracy = mean(accuracy);
disp(['Average Accuracy: ', num2str(averageAccuracy)]);

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
figure;
heatmap(meanConfMatrix, 'Colormap');
title('Mean Normalized Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');

% Optional: Display the values on the heatmap
textStrings = num2str(meanConfMatrix(:), '%0.2f');       % Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));             % Remove any space padding
[x, y] = meshgrid(1:numClasses);                         % Create x and y coordinates for the strings

%{
 hStrings = text(x(:), y(:), textStrings(:), ...          % Plot the strings
                'HorizontalAlignment', 'center');
colorbar; 
%}


















% Retrain the model
newModel = fitclinear(combinedX, combinedY, ...
                      'ClassNames', classNames, ...
                      'Prior', prior, ...
                      'Cost', cost, ...
                      'Lambda', lambda, ...
                      'Regularization', 'ridge', ...
                      'Learner', learner, ...
                      'ObservationsIn', 'rows');

% Once the model is trained, you can make predictions as before
% [predictedLabels, scores] = predict(newModel, newValidationDataX);


% do kfold cross validation








kernelFunction = mdl.ClassificationKernel.KernelParameters.Function;
boxConstraint = mdl.ClassificationKernel.BoxConstraints;

% ...any other parameters

% Prepare your combined data
% Assuming X is your predictor matrix and Y is your response vector
combinedX = [originalX; augmentedX];
combinedY = [originalY; augmentedY];

% Create a new SVM template with the same parameters as the old model
% (This is assuming a lot about your model that might not be correct)
% You'd replace 'rbf' with whatever kernelFunction you have extracted
template = templateSVM(...
    'KernelFunction', kernelFunction, ...
    'BoxConstraint', boxConstraint);
    ...any other parameters...);

% Train the new model on the combined dataset
newModel = fitcecoc(combinedX, combinedY, 'Learners', template);

% Validate your new model
% Make predictions using the new model
[newPredictions, newScores] = predict(newModel, validationX);

% Then compare these predictions to your validationY to assess the performance








% Compute PCA on training data
pca_d = pca(d_z', "NumComponents",2, 'Algorithm','svd', 'Centered',false,'Weights', norm_estimates);

%
% figure, gscatter(pca_d(:,1), pca_d(:,2), response)



T_pred_to_use = T_pred;
pred= table2array(T_pred_to_use(:, pred_names));

% % impute using KNN
imputed = knnimpute(pred');
imputed = imputed';
is_nan_idx = false(size(imputed,1),1);
% write down the number back to the table
% this_T = [not_best,array2table(imputed,'VariableNames',best_predictors)];
d = imputed;

%% << HERE :  DO data augmentation >> %% 



% Alternatively, remove rows with nan
% is_nan_idx = sum(isnan(pred),2) >0;
% d = pred(~is_nan_idx,:);
% Do PCA
pred_z = zscore(d);
pred_pca = do_pca_gini(pred_z, norm_estimates);
% pred_pca = pca(pred_z', "NumComponents",2, 'Algorithm','svd', 'Centered',false,'Weights', best_pred_vals);
% figure, scatter(pred_pca(:,1), pred_pca(:,2))

%% predict with model
[yfit,~] = mdl.predictFcn(pred_pca);
figure, gscatter(pred_pca(:,1), pred_pca(:,2), yfit, [],'o', 'filled')
title('LR')
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

