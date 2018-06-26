%% Lines to edit
fnIJ='/Applications/MATLAB_R2018a.app/java/jar/ij.jar';     % Path to "ij.jar"
fnMIJ='/Applications/MATLAB_R2018a.app/java/jar/mij.jar';   % Path to "mij.jar"

%%
javaaddpath(fnIJ);
javaaddpath(fnMIJ);
addpath('code');
addpath('code/external/utils');
addpath(genpath('code/external/InvPbLib'));
