function [ L0_Trees, L1_Trees, L2_Trees ] = TrainWithExpOnly( expMat, accMat )
%TRAINWITHEXPONLY
%   Train classifiers as described in the readme
%   expMat and accMat are asssumed to matrices loaded as described in
%   step 1 of the readme.

[r,c] = size(expMat);

%   Step 2: Create empirical reference distributions for gene expression
%   values associated to accessible and inaccessible regions
oneDist = expMat(accMat==1);
%Just for generalizability, not necessary for the example data
oneDist(isnan(oneDist)) = [];

zeroDist = expMat(accMat==0);
%Just for generalizability, not necessary for the example data
zeroDist(isnan(zeroDist)) = [];

%   Step 3: Generate matrices with mahalanobis distances to 'oneDist' and
%   'zeroDist' respectively. Need to loop through the columns due to the
%   implementation of the built-in 'mahal' function.
mahal_OneDist = zeros(r,c);
mahal_zeroDist = zeros(r,c);

for i=1:c
    mahal_OneDist(:,i) = mahal(expMat(:,i),oneDist);
    mahal_zeroDist(:,i) = mahal(expMat(:,i),zeroDist);
end

%   Step 4: Create cell array for training input. Importantly, remove NaN's
%   from each matrix.
trainingInp = cell(c,1);
for i = 1:c
   %find all values that are not NaN's in i-th column of 'expMat'.
   idx = find(~isnan(expMat(:,i)));
   trainingInp{i} = [expMat(idx,i) mahal_zeroDist(idx,i) mahal_OneDist(idx,i) accMat(idx,i)];
end

%   Step 5: Call function 'TrainAllClassifiers.m' with training input data
[ L0_Trees, L1_Trees, L2_Trees ] = TrainAllClassifiers( trainingInp );



end

