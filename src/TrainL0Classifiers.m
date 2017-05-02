function [ Trees_AllExamples ] = TrainL0Classifiers(expression)
%TRAINFORALLEXAMPLES Summary of this function goes here
%   Detailed explanation goes here

Trees_AllExamples = cell(numel(expression),1);
parfor exp=1:numel(expression)
    display(strcat('L0 classifier ',num2str(exp),' trained'));
    BaggedTrees = cell(1000,1);
    for i=1:1000
        y = datasample(expression{exp},1000,'Replace',false);
        c = fitctree(y(:,1:end-1),y(:,end),'MaxNumSplits',20);
        c = compact(c);
        BaggedTrees{i} = c;
    end
    Trees_AllExamples{exp} = BaggedTrees;
end

end

