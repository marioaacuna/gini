function [d, s] = do_pca_gini(X, best_vals)
% this will just do PCA as we have been doing with good results
[d, s] = pca(X', "NumComponents",2, 'Algorithm','svd', 'Centered',false,'Weights', best_vals);

end
