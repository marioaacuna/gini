function GC = general_configs()
    GC = struct();
    %%
    main_root = 'M:\Mario';
    GC.data_root_path = [main_root,'\Gini'];
    GC.raw_data_folder = [main_root,'\Gini\Data']; % to be changed if table is placed somewhere else
    GC.plot_path = [main_root,'\Gini\plots'];
    % Get path of this file
    current_path = mfilename('fullpath');
    % Remove filename to get root path of the repository
    repository_root_path = regexp(current_path, filesep(), 'split');
    GC.repository_root_path = fullfile(repository_root_path{1:end-1});
    GC.toolboxes_root_path = fullfile(GC.repository_root_path, 'Code', '3rd party toolboxes');
    GC.forbidden_folders = {'Superuser', '\.', '3rd party toolboxes', 'Documentation', 'python', '_test'};  % This list will be passed to regexp (make sure special characters are properly escaped)
    % Python
    GC.python = struct();
    GC.python.environment_name = 'env_ca_imaging';
    [~, msg] = system(sprintf('activate %s && python -c "import sys; print(sys.executable)"', GC.python.environment_name));
    GC.python.interpreter_path = msg(1:end-1);
    GC.python.scripts_path = fullfile(GC.repository_root_path, 'Code', 'python');
    % For data analysis of CFA data
    GC.threshold_depth_L5 = 300;
    GC.variables_to_discard = { 'Date', 'Slice', 'ID', 'Burst'}; % , 'Burst', 'ICAmp'
    GC.variables_to_evaluate = {'Depth', 'Bifurcation', 'Polarity','Perimeter','Area','Diameter', 'Measured',...
                            'MaxH', 'MaxV','Den', 'MaxH_1', 'MaxV_1', 'Angle', 'MaxOrder',  'Oblique', 'SAG',...
                             'SAGAmp', 'InputR', 'RMP', 'APThreshold', 'Tau', 'Adaptation', 'APAmplitude', 'APHW',...
                             'VeloDepo', 'VelRepo','ICAmp',  'Burst'};
    GC.init_conds.CFA = {'Saline', 'CFA'};
    GC.init_tps.CFA = {'d1', 'd7', 'd7NS'};

end