function [u_midway_1,u_midway_2]=midway(u_1,u_2)

u_midway_1 = zeros(size(u_1));
u_midway_2 = zeros(size(u_2));

[u_sort_1 ,index_u_1] = sort(u_1 (:));
[u_sort_2 ,index_u_2] = sort(u_2 (:));
u_midway_1(index_u_1) = (u_sort_1 + u_sort_2 )/2;
u_midway_2(index_u_2) = (u_sort_1 + u_sort_2 )/2;

u_midway_1 = reshape(u_midway_1 ,size(u_1));
u_midway_2 = reshape(u_midway_2 ,size(u_2));
