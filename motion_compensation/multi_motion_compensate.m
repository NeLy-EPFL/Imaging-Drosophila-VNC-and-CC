function multi_motion_compensate(fnInput1,fnInput2,fnMatch,fnDeepMatching,N,param,outdir,lambdas,gammas)
%% Description
% Motion is estimated with a brigthness constancy data term defined on fnInput1 
% and a feature matching similarity term defined on fnMatch. The sequences fnIput1 and
% fnInput2 are warped according to the estimated motion field.
% For more details see the paper: "Imaging neural activity in the ventral
% nerve cord of behaving adult Drosophila", bioRxiv
%
%% Input
% fnInput1: filename of the  sequence used for the brightness constancy term, in TIF format
% fnInput2: filename of the sequence warped with the motion field estimated from fnInput1 and fnMatch, in TIF format
% fnMatch: filename of the sequence used for feature matching, in TIF format
% fnDeepMatching: filename of the deepmatching code
% N: number of frames to process
% param: parameters of the algorithm (see 'default_parameters.m')
% outdir: list of output directories for different combination of lambda and gamma
% lambdas: list of lambda parameters
% gammas: list of gamma parameters
%
%     Copyright (C) 2017 F. Aymanns, florian.aymanns@epfl.ch
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.

start_time = datestr(now,'yyyymmddHHMM');
tmpdir = ['tmp',start_time];
mkdir(tmpdir);

% Input data
seq=mijread_stack(fnInput1);
seqW=mijread_stack(fnInput2);
seqMatch=mijread_stack(fnMatch);

if N==-1
    N=size(seq,3);
end

seq=double(seq(:,:,1:N));
seqRescale=(seq-min(seq(:)))/(max(seq(:))-min(seq(:)))*255;
seqW=double(seqW(:,:,1:N));
seqMatch=double(seqMatch(:,:,1:N));
seqMatch=(seqMatch-min(seqMatch(:)))/(max(seqMatch(:))-min(seqMatch(:)))*255;

% Motion estimation
i1Match=seqMatch(:,:,1);
tic;
parfor t=1:N-1 % Replace parfor by for if you don't want to parallelize
    presteps_to_compute_motion(i1Match,seqMatch(:,:,t+1),t,tmpdir);
end
fprintf('Presteps_to_compute_motion done!\n');
fprintf(['Time for presteps computation ',num2str(toc),'\n']);
return_number_of_running_processes = 'ps r | wc -l';
tic;
for t=1:N-1
    [status, n_processes] = system(return_number_of_running_processes);
    while str2num(n_processes)-1 > 28
        pause(1);
        [status, n_processes] = system(return_number_of_running_processes);
    end
    fprintf(['Strating deep of frame ', num2str(t),'\n']);
    fnI1=[tmpdir,'/I1_',num2str(t),'.png'];
    fnI2=[tmpdir,'/I2_',num2str(t),'.png'];
    fnMatch=[tmpdir,'/match_',num2str(t),'.txt'];
    command = [fnDeepMatching,'/deepmatching ',fnI1,' ',fnI2,' -out ',fnMatch,' &'];
    s = system(command);
end
fprintf(['Time for deepmatching for loop ',num2str(toc),'\n']);

lxg = length(lambdas)*length(gammas);
i1=seqRescale(:,:,1);
for t = 1:N-1
    i2=seqRescale(:,:,t+1);
    [i10,i2]=midway(i1,i2);

    w = poststeps_to_compute_motion(i10,i2,i1Match,param,t,tmpdir,lambdas,gammas);
    for idx = 1:lxg
        dlmwrite([char(outdir(idx)),'/wx_frame',num2str(t),'.dat'],w(:,:,1,idx));
        dlmwrite([char(outdir(idx)),'/wy_frame',num2str(t),'.dat'],w(:,:,2,idx));
    end
    fprintf(['poststeps_to_compute_motion done for frame ',num2str(t),'\n']);
end

lxg = length(lambdas)*length(gammas);
for idx = 1:lxg
    tic;

    colorFlow=zeros(size(seqRescale,1),size(seqRescale,2),3,size(seqRescale,3)-1);
    vectorFlow=zeros(512,512,size(seqRescale,3)-1);
    seqWarped=seq;
    seqwWarped=seqW;

    for t = 1:N-1
        wx = load([char(outdir(idx)),'/wx_frame',num2str(t),'.dat']);
        wy = load([char(outdir(idx)),'/wy_frame',num2str(t),'.dat']);
        colorFlow(:,:,:,t) = flowToColor(cat(3,wx,wy));
        vectorFlow(:,:,t) = flowToVectorImage(cat(3,wx,wy),[18,18],[512,512],idx*10^N+t,tmpdir);
        seqWarped(:,:,t+1)=warpImg(seq(:,:,t+1),wx,wy);
        seqwWarped(:,:,t+1)=warpImg(seqW(:,:,t+1),wx,wy);
    end 

    [cY1, mY1, ng1] = motion_metrics(seqWarped);
    [cY2, mY2, ng2] = motion_metrics(seqwWarped);

    fnOut1=[char(outdir(idx)),'/warped1.tif'];              % Sequence fnIn1 warped
    fnOut2=[char(outdir(idx)),'/warped2.tif'];              % Sequence fnIn2 warped
    fnColor=[char(outdir(idx)),'/color_flow.tif'];           % Color visualization of the motion field
    fnVector=[char(outdir(idx)),'/vectors.tif'];

    mijwrite_stack(single(seqWarped),fnOut1);
    mijwrite_stack(single(seqwWarped),fnOut2);
    mijwrite_stack(single(colorFlow),fnColor,1);
    mijwrite_stack(single(vectorFlow),fnVector);

    dlmwrite([char(outdir(idx)),'/cY_Out1.dat'],cY1);
    dlmwrite([char(outdir(idx)),'/cY_Out2.dat'],cY2);
    dlmwrite([char(outdir(idx)),'/mY_Out1.dat'],mY1);
    dlmwrite([char(outdir(idx)),'/mY_Out2.dat'],mY2);
    dlmwrite([char(outdir(idx)),'/ng_Out1.dat'],ng1);
    dlmwrite([char(outdir(idx)),'/ng_Out2.dat'],ng2);

    fprintf(['parfor loop over idx ',num2str(idx),' finished in ',num2str(toc),'\n']);
end
rmdir(tmpdir);
end

function presteps_to_compute_motion(I1Match,I2Match,t,tmpdir)

fnI1=[tmpdir,'/I1_',num2str(t),'.png'];
fnI2=[tmpdir,'/I2_',num2str(t),'.png'];
imwrite(uint8(I1Match),fnI1);
imwrite(uint8(I2Match),fnI2);

end

function w = poststeps_to_compute_motion(I1,I2,I1Match,param,t,tmpdir,lambdas,gammas)

sz0 = size(I1);
I1=padarray(I1,[15,15],'replicate');
I2=padarray(I2,[15,15],'replicate');

fnI1=   [tmpdir,'/I1_',num2str(t),'.png'];
fnI2=   [tmpdir,'/I2_',num2str(t),'.png'];
fnMatch=[tmpdir,'/match_',num2str(t),'.txt'];

formatSpec = '%u %u %u %u %f %u';
sizeCorresp = [6 Inf];

while ~exist(fnMatch,'file')
  pause(1);
end
f=fopen(fnMatch);
corresp = fscanf(f,formatSpec,sizeCorresp);
command = ['rm ',fnI1]; s = system(command);
command = ['rm ',fnI2]; s = system(command);
command = ['rm ',fnMatch]; s = system(command);

thresh=param.threshMatch;
Iseg=segment(I1Match,'variance',thresh);

matches = zeros(5,1);
k=0;
for i=1:size(corresp,2)
    if corresp(1,i)>0 && corresp(1,i)<=sz0(2) && corresp(2,i)>0 && corresp(2,i)<=sz0(1)
        if Iseg(corresp(2,i),corresp(1,i)) == 1
            k = k+1;
            matches(1,k) = corresp(1,i);
            matches(2,k) = corresp(2,i);
            matches(4,k) = corresp(4,i)-corresp(2,i);
            matches(3,k) = corresp(3,i)-corresp(1,i);
        end
    end
end

scores = corresp(5,:);
idx = find(corresp(5,:));
scores = scores(scores~=0);
scores = (scores-min(scores))/(max(scores)-min(scores));
for j=1:length(idx)
    matches(5,idx(j)) = scores(j);
end

%% Optical flow
print = 0;  % if print =1:verbose mode
disp = 0;   % if disp=1; displays the results at each iteration
tmp_param = param;
idx=1;
w = zeros(sz0(1),sz0(2),2,length(lambdas)*length(gammas));
for l = lambdas
    for g = gammas
        tmp_param.lambda = l;
        tmp_param.gamma = g;
        tmp_w = of_l1_l2_fm_admm(I1,I2,sz0,matches,tmp_param,print,disp);
        w(:,:,:,idx) = crop_fit_size_center(tmp_w,[sz0(1),sz0(2),2]);
        idx = idx+1;
    end
end
end
