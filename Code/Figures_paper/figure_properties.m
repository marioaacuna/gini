function FP = figure_properties()

FP = struct();

% Fonts
FP.font1.family           = 'Arial';
FP.font1.size.suptitle    = 18;
FP.font1.size.title       = 16;
FP.font1.size.tick_labels = 8;
FP.font1.size.axes_labels = 12;

% Colors - groups
FP.colors.basic.red        = [1, 0, 0];
FP.colors.basic.blue       = [.05, .48, .75];
FP.colors.basic.orange     = [1, .65, 0];
FP.colors.basic.green      = [0, .8, 0];
FP.colors.basic.gray       = [.7, .7, .7];
FP.colors.basic.light_gray = [.9, .9, .9];
FP.colors.basic.black      = [0, 0, 0];


% Colors - groups
FP.colors.groups.CCI   = [1, 0, 0];
FP.colors.groups.sham  = [.05, .48, .75];
FP.colors.groups.naive = [1, .65, 0];
FP.colors.groups.SNI   = [1, 0, 0];
FP.colors.groups.CFA   = [1, 0, 0];
FP.colors.groups.SAL   = [.05, .48, .75];

FP.colors.groups.a   = [1, .65, 0];
FP.colors.groups.b   = [0, 1, 0];


% Colors - Clusters
FP.colors.clusters = [1,.5,0.1;...
                     1,0,1;...
                     0,1,1;...
                     1,0,0;...
                     0,1,0;...
                     0,0,1;...
                    .7,.7,.7;...
                     0.2,.7,.2;...
                     0,0.5,.5];

% Scatterplots
FP.scatterplots.size_small = 7;
FP.scatterplots.size_large = 10;


