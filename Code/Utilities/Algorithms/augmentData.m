function augmented_data = augmentData(original_data, noise_factors)
    % original_data: The original dataset (a matrix where rows are samples and columns are features)
    % noise_factors: A vector containing noise factors for each feature

    % Preallocate the augmented_data array
    augmented_data = zeros(size(original_data));
    
    % Loop over each feature (column)
    for feature_idx = 1:size(original_data, 2)
        % Extract the current feature column
        feature_data = original_data(:, feature_idx);
        
        % Calculate the standard deviation of the current feature data
        feature_std = std(feature_data);
        
        % Generate random Gaussian noise based on the feature's standard deviation
        % and the specified noise factor for this feature
        noise = randn(size(feature_data)) * feature_std * noise_factors(feature_idx);
        
        % Add the noise to the original data to get the augmented data for this feature
        augmented_data(:, feature_idx) = feature_data + noise;
    end
end
