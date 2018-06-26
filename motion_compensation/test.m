close all;
clear all;
warn = warning ('off','all');
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end
pctRunOnAll warning('off','all')

%% Parameters
param = default_parameters();
param.lambda = 1000;    % Regularization parameter
param.gamma = 100;      % Sets the strength of the feature matching constraint

%% Input - Output
type=0;                                     % 0 for Mac users, 1 for Linux users
fnMatch='data/1_Horizontal_VNC/tdTom.tif';  % Sequence used for the feature matching similarity term
fnIn1='data/1_Horizontal_VNC/tdTom.tif';     % Sequence used for the brightness constancy term 
fnIn2='data/1_Horizontal_VNC/GC6s.tif';    % Sequence warped with the motion field estimated from fnIn1 and fnMatch
N=3;                                        % Motion is estimated on frames 1 to N. If N=-1, motion is estimated on
                                            % the whole sequence
fnSave='results/test/';                     % Folder where the results are saved
mkdir(fnSave);

fnOut1=[fnSave,'warped1.tif'];              % Sequence fnIn1 warped
fnOut2=[fnSave,'warped2.tif'];              % Sequence fnIn2 warped
fnColor=[fnSave,'colorFlow.tif'];           % Color visualization of the motion field

if type==0
    fnDeepMatching='code/external/deepmatching_1.2.2_c++_mac';      
elseif type==1
    fnDeepMatching='code/external/deepmatching_1.2.2_c++_linux';      
end
addpath(fnDeepMatching);

%% Perform motion compensation
motion_compensate(fnIn1,fnIn2,fnMatch,fnDeepMatching,fnOut1,fnOut2,fnColor,N,param);

%% Perform parallelized motion compensation for multiple lambdas and gammas
% Uncomment this section if you want to compute perform the registration
% for multiple lambda and gamm values. In addition to the output of
% motion_compensate, it saves images of the vector fields and several
% metrics to judge registration quality:
%   cY = correlation coefficient of each frame with the mean
%   mY = mean image
%   ng = norm of gradient of mean image

% lambdas = [500,1000];
% gammas = [0,10];
% outdir = cell(length(lambdas)*length(gammas));
% idx=1;
% for l = lambdas
%     for g = gammas
%         outdir(idx) = cellstr([fnSave,'/l',num2str(l),'g',num2str(g)]);
%         mkdir(char(outdir(idx)));
%         idx = idx + 1;
%     end
% end
% multi_motion_compensate(fnIn1,fnIn2,fnMatch,fnDeepMatching,N,param,outdir,lambdas,gammas)