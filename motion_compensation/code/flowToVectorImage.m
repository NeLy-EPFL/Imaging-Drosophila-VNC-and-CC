function A = flowToVectorImage(w, n_sampling_points, resolution, frame_number, tmpdir)
%% Description
% This function returns the matrix representation of an image showing a
% down sampled version of the vector field w.
%
%% Input
% w is flow of form n x n x 2
% n_sampling_points is the number of sampling point in x and y direction 
% resolution is an array of length 2 which specifies the resolution of the resulting vector image
% frame_number is needed for parallel execution to separate tmp files
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



    sampling_rate_1 = round(size(w,1)/n_sampling_points(1));
    sampling_rate_2 = round(size(w,1)/n_sampling_points(2));
    u = w(1:sampling_rate_1:end,1:sampling_rate_1:end,1);
    v = w(1:sampling_rate_2:end,1:sampling_rate_2:end,2);
    [x,y] = meshgrid(1:sampling_rate_2:size(w,2),1:sampling_rate_1:size(w,1));
    figure('visible','off');
    quiver(x,y,u,v,'color','k','AutoScale','off','LineWidth', 8);
    set(gca,'XTick',[]); % Remove the ticks in the x axis!
    set(gca,'YTick',[]); % Remove the ticks in the y axis
    set(gca,'XLim',[1,size(w,1)]);
    set(gca,'YLim',[1,size(w,2)]);
    set(gca,'Visible','off')
    set(gcf,'PaperUnits','inches','PaperPosition',[0 0 resolution(1) resolution(2)]);
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset;
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];

    filename = fullfile(tmpdir,['tmpvector',num2str(frame_number),'.tiff']);
    print(filename,'-dtiff','-r1');
    im = imread(filename);
    A = imerode(imadjust(rgb2gray(im),[0.99, 1]), strel('square',1));

    delete(filename);
