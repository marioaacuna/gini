function [cvs, noise_factors] = calculateCVandNoiseFactors(data, noise_factor_fraction)
    % Calculate the mean and standard deviation for each feature
    means = mean(data);
    std_devs = std(data);

    % Calculate the coefficient of variation (CV) for each feature
    cvs = std_devs ./ means;

    % Apply the noise_factor_fraction to calculate noise factors
    noise_factors = cvs * noise_factor_fraction;

    % Display the results
    disp('Coefficient of Variation (CV) for each feature:');
    disp(cvs);
    disp('Suggested noise factors for each feature:');
    disp(noise_factors);
end
