function param = default_parameters()

param = struct('lambda',10000,'gamma',100,'mu',0.1,'nu',0.1,'c2fSpacing',1.5,'threshMatch',70,...
    'occThresh',0.5,'consistencyTol',2,'changeTol',1e-3,'nbWarps',1,'sigmaS',1,'sigmaC',200,'maxIters',150,'iWM',50);
