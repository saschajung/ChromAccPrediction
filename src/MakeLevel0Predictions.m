function [ L0Predictions ] = MakeLevel0Predictions( Trees_L0, expression )
%MAKELEVEL0PREDICTIONS Summary of this function goes here
%   Detailed explanation goes here
L0Predictions = z;
for tree=1:numel(Trees_L0)
    L0Prediction = [];
    for subtree=1:numel(Trees_L0{tree})
        [~,s] = predict(Trees_L0{tree}{subtree},expression);
        L0Prediction = [L0Prediction s(:,2)];
    end
    L0Predictions = [L0Predictions;L0Prediction];
end

end

