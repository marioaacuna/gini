function test_clustering_morpho_phys(type_experiment)
%% Preamble
% Test for evaluating morphological and electrophysiological features of
% neurons in the L5 ACC. 

global GC
clc
close all
warning off
%% Get the data
save_figs = 1;
% take_only_saline = 1;
FP = figure_properties();
%%
% type_dim_red = 'UMAP';
type_dim_red = 't-SNE';
% type_dim_red = 'PCA';

% addpath(GC.raw_data_folder)
input_file = os.path.join(GC.raw_data_folder, 'input.xlsx');
% input_file =  'M:\Mario\Gini\dataset.xlsx'; % from 2019 dataset
opts = detectImportOptions(input_file);

parameters = opts.VariableNames;
variables_to_discard = GC.variables_to_discard;%{'Date', 'Slice', 'ID', 'Burst'}; % , 'Burst', 'ICAmp'
Variables = parameters(~ismember(parameters, variables_to_discard)); 
% Variables = parameters(ismember(parameters, {'SAG', 'Diameter'}));

opts.SelectedVariableNames = Variables;

T = readtable( input_file,opts);
opts.SelectedVariableNames = 'Label';
% L = readtable(input_file,opts);
% data_csv = readmatrix('M:\Mario\Gini\dataset_Ma.csv');
threshold = GC.threshold_depth_L5; % Thomas' paper
T(T.Depth < threshold, :) =[];

init_conds = GC.init_conds.CFA;
init_tps = GC.init_tps.CFA;

if strcmp(type_experiment, 'S') 
    take_only_saline = 1;
    init_conds = init_conds(1);
    in_idx = startsWith(T.Experiment, init_conds);
    T = T(in_idx,:);

elseif strcmp(type_experiment, 'C') 
    take_only_cfa = 1;
    init_conds = init_conds(2);
    in_idx = startsWith(T.Experiment, init_conds);
    T = T(in_idx,:);
else
%     init_conds = init_conds;
    in_idx = startsWith(T.Experiment, init_conds);
    T = T(in_idx,:);
end
T_new = T;
original_labels = T_new.Label;
% response = categorical(original_labels);
response = double(ismember(original_labels, {'b'}));


% Run a tree classifier to determine the important variables
tree_1 = fitrtree(T_new(:,~ismember(T_new.Properties.VariableNames, {'Label', 'Experiment'})), response,'PredictorSelection','curvature','Surrogate','on');
% tree_1 = fitrtree(T, response);

% view(tree_1.Trained{10},'Mode','graph')
% view(tree_1,'Mode','graph')
table_2_csv = T_new(:,~ismember(T_new.Properties.VariableNames, {'Label', 'Experiment'}));
data_csv = table2array(table_2_csv);

%%
imp = predictorImportance(tree_1);
%
figure_predictors = figure('Color','w');
bar(imp);
title('Predictor Importance Estimates');
ylabel('Estimates');
xlabel('Predictors');
h = gca;
xticks([1:length( tree_1.PredictorNames)])
h.XTickLabel = tree_1.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';
if save_figs
    fig_filename = os.path.join(GC.plot_path, type_experiment,'fig_predictors.pdf');
%     print -dpdf -painters fig_filename
%     export_fig(fig_filename, figure_predictors); close(figure_predictors);
% hgexport(figure_predictors,fig_filename); %Set desired file name
    saveas(figure_predictors,fig_filename)
    close(figure_predictors)
end
%%
% remove NaNs
[rows, ~]= find(isnan(data_csv));
data_csv(unique(rows),:)=[];
original_labels(unique(rows),:)=[];

%% Set parameters
metric ='correlation'; %'correlation'; % euclidean
n_components = 2;%n_stimuli+1

%% Run dim red and clustering
switch type_dim_red
    case 'UMAP'
        n_neighbors = 3 ;%n_stimuli + 1; 3
        min_dist =0.9;% 0.9
        random_state = 42;
        csv_output_filename = 'M:\Mario\Gini\Code\_temp\dataset_output.csv';
        csv_input_filename='M:\Mario\Gini\Code\_temp\dataset_to_UMAP.csv';


        first_row = zeros(1,size(data_csv,2));
        csvwrite(csv_input_filename, [first_row;data_csv])

        run_in_python('dim_reduction_UMAP', ['input_filename=', csv_input_filename], ['output_filename=', csv_output_filename],['n_neighbors=' ,num2str(n_neighbors)], ['min_dist=', num2str(min_dist)], ['metric=', metric], ['n_components=', num2str(n_components)], ['random_state=', num2str(random_state)] ) %
        
        data_to_clusters = csvread(csv_output_filename);
        delete(csv_input_filename, csv_output_filename)
        
    case 't-SNE'
        exaggeration = 4; perplexity = 5; learn_rate = 500;
        [data_to_clusters, loss] = tsne(data_csv, 'Algorithm','exact', 'Distance',metric,...
            'NumDimensions',n_components, 'NumPCAComponents',2, ...
            'Standardize',1, 'Verbose',0, 'Exaggeration',exaggeration, 'Perplexity',perplexity, 'LearnRate', learn_rate);
    case 'PCA'
        [coeff, data_to_clusters, latent, tsquared, explained, mu] = pca(data_csv, 'Algorithm', 'eig', 'NumComponents', 2);
%     case 'k-means'
%         [idx, data_to_clusters] = kmeans(data_csv, 2, 'Distance', metric);
%         data_to_clusters = data_to_clusters(:,1:2)';
end
%% Plot Indexes

figure_dim_red = figure('color', 'w', 'pos',[100, 100, 600, 600]);
h = gscatter(data_to_clusters(:,1), data_to_clusters(:,2), original_labels) ; title(type_dim_red);
names = {h.DisplayName};

is_b = strcmp(names, 'b');
set(h(is_b), 'Color', FP.colors.groups.b)
set(h(~is_b), 'Color', FP.colors.groups.a)
labelsx= [type_dim_red,'-1'];
labelsy= [type_dim_red,'-2'];
xlabel(labelsx)
ylabel(labelsy)
axis square

if save_figs
     fig_filename = os.path.join(GC.plot_path, type_experiment,'fig_dim_red.pdf');  
     saveas(figure_dim_red,fig_filename)
     close(figure_dim_red)

end
    
    
    
% [~,C] = kmeans(data_to_clusters,2)
%%
%%
num_important_features= sum(imp>1*std(imp));
[~, idx] = maxk(imp, num_important_features);
predictors = tree_1.PredictorNames(idx);
opts2 = detectImportOptions(input_file);
opts2.SelectedVariableNames = predictors;

T_new_round = T_new(:,predictors);
% data_csv_new = table2array(T_new);

% [rows, ~]= find(isnan(data_csv_new));
% data_csv_new(unique(rows),:)=[];
% original_labels(unique(rows),:)=[];

data_csv_new = table2array(T_new_round);
data_csv_new(rows,:)=[];

%% Run dim red and clustering
switch type_dim_red
    case 'UMAP'
        n_neighbors = 3 ;%n_stimuli + 1; 3
        min_dist =0.9;% 0.9
        random_state = 42;

        first_row = zeros(1,size(data_csv_new,2));
        csvwrite(csv_input_filename, [first_row;data_csv_new])

        run_in_python('dim_reduction_UMAP', ['input_filename=', csv_input_filename], ['output_filename=', csv_output_filename],['n_neighbors=' ,num2str(n_neighbors)], ['min_dist=', num2str(min_dist)], ['metric=', metric], ['n_components=', num2str(n_components)], ['random_state=', num2str(random_state)] ) %
        
        data_to_clusters2 = csvread(csv_output_filename);
        delete(csv_input_filename, csv_output_filename)
        
    case 't-SNE'
        exaggeration = 1; perplexity = 5; learn_rate = 500;
        [data_to_clusters2, loss] = tsne(data_csv_new, 'Algorithm','exact', 'Distance',metric,...
            'NumDimensions',n_components, 'NumPCAComponents',2, ...
            'Standardize',1, 'Verbose',0, 'Exaggeration',exaggeration, 'Perplexity',perplexity, 'LearnRate', learn_rate);
    case 'PCA'
        [coeff, data_to_clusters2, latent, tsquared, explained, mu] = pca(data_csv_new, 'Algorithm', 'svd', 'NumComponents', 2);
%     case 'k-means'
%         [idx, data_to_clusters] = kmeans(data_csv, 2, 'Distance', metric);
%         data_to_clusters = data_to_clusters(:,1:2)';
end


%%

fig_dim_red_imp_fea = figure('color', 'w','pos',[100, 100, 600, 600]);
h = gscatter(data_to_clusters2(:,1), data_to_clusters2(:,2), original_labels); title(type_dim_red)
names = {h.DisplayName};
is_b = strcmp(names, 'b');
set(h(is_b), 'Color', FP.colors.groups.b)
set(h(~is_b), 'Color', FP.colors.groups.a)

labelsx= [type_dim_red,'-1'];
labelsy= [type_dim_red,'-2'];
xlabel(labelsx)
ylabel(labelsy)
axis square
title(['Only taking ', num2str(num_important_features), ' important features'])

if save_figs
     fig_filename = os.path.join(GC.plot_path, type_experiment,'fig_dim_red_only_important_f.pdf');  
     saveas(fig_dim_red_imp_fea,fig_filename)
     close(fig_dim_red_imp_fea)
end


%% Classification analysis
disp('done')

end