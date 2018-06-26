function motion_compensate(fnInput1,fnInput2,fnMatch,fnDeepMatching,fnOut1,fnOut2,fnColor,N,param)
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
% fnOut1: filename used to save the warped version of fnInput1, in TIF format
% fnOut2: filename used to save the warped version of fnInput2, in TIF format
% fnOut2: filename used to save the color visualization of the estimated motion, in TIF format
% N: number of frames to process
% param: parameters of the algorithm (see 'default_parameters.m')
%
%     Copyright (C) 2017 D. Fortun, denis.fortun@epfl.ch
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

%% Input data
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

%% Motion estimation
w=zeros(size(seqRescale,1),size(seqRescale,2),2,size(seqRescale,3)-1);
colorFlow=zeros(size(seqRescale,1),size(seqRescale,2),3,size(seqRescale,3)-1);
i1=seqRescale(:,:,1);
i1Match=seqMatch(:,:,1);
for t=1:N-1 % Replace parfor by for if you don't want to parallelize
    fprintf('Frame %i\n',t);
    i2=seqRescale(:,:,t+1);
    i2Match=seqMatch(:,:,t+1);

    [i10,i2]=midway(i1,i2);

    w(:,:,:,t) = compute_motion(i10,i2,i1Match,i2Match,fnDeepMatching,param,t);
    colorFlow(:,:,:,t)=flowToColor(w(:,:,:,t));
end
                
%% Registration
seqWarped=seq;
seqwWarped=seqW;
parfor t=1:N-1
    seqWarped(:,:,t+1)=warpImg(seq(:,:,t+1),w(:,:,1,t),w(:,:,2,t));
    seqwWarped(:,:,t+1)=warpImg(seqW(:,:,t+1),w(:,:,1,t),w(:,:,2,t));
end

%% Save
mijwrite_stack(single(seqWarped),fnOut1);
mijwrite_stack(single(seqwWarped),fnOut2);
mijwrite_stack(single(colorFlow),fnColor,1);
