function set_paths()
    % Set global variables
    global GC

    % Read general_configs
    GC = general_configs();

    % Make sure server is reachable
    if ~exist(GC.data_root_path, 'dir')
        error('launcher:unreachable_server', 'Server is unreachable, please retry running the script.\nIf the problem persists, check that ''data_root_path'' in general_configs.m is correct.')
    end
    
    % Disable warning
    warning('off','MATLAB:rmpath:DirNotFound')
    % Get list of folders in repository, and remove them all from MATLAB path
    all_folders = genpath(GC.repository_root_path);
    rmpath(all_folders)
    % Split folder names
    if ispc(), sep=';'; else, sep=':'; end
    all_folders = regexp(all_folders, sep, 'split');
    % Make sure that forbidden folders are not added to MATLAB's path
    forbidden_folders = [GC.forbidden_folders];
    bad_folders_idx = cell2mat(cellfun(@(x) ~isempty(regexp(x,strjoin(forbidden_folders,'|'),'once')), all_folders, 'UniformOutput',false));
    good_folders = strjoin(all_folders(~bad_folders_idx), ';');
    % Add good folders and remove bad ones
    addpath(good_folders)
    warning('on','MATLAB:rmpath:DirNotFound')
    
    % Make sure that files get deleted
    recycle('off')
    % Clear command window
    clc
    disp('all set')

%     % Initialize connection to SQL database, if not previously done in this MATLAB
%     % session. Do not attempt to close this connection or MATLAB will force quit!
%     if isempty(SQL_DATABASE_connection)
%         SQL_DATABASE_connection = SQL_database.initialize_db_connection();
%     end
% 
%     % Run launcher_GUI
%     launcher_GUI();

%% Mlint exceptions
%#ok<*CLFUNC>
