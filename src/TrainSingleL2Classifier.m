function [trainedClassifier, validationAccuracy] = TrainSingleL2Classifier(trainingData,crossval)
% Convert input to table
inputTable = table(trainingData);
inputTable.Properties.VariableNames = {'column'};

allVars = cell(1,numel(trainingData(1,:)));
predictorVar = cell(1,numel(trainingData(1,:))-1);
responseVar = numel(trainingData(1,:));

for i=1:(numel(trainingData(1,:))-1)
   allVars{i} = strcat('column_',num2str(i));
   predictorVar{i} = strcat('column_',num2str(i));
end

allVars{responseVar} = strcat('column_',num2str(responseVar));

% Split matrices in the input table into vectors
inputTable = [inputTable(:,setdiff(inputTable.Properties.VariableNames, {'column'})), array2table(table2array(inputTable(:,{'column'})), 'VariableNames', allVars)];

% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.
predictorNames = predictorVar;
predictors = inputTable(:, predictorNames);
response = inputTable(:,responseVar);

% Train a classifier
% This code specifies all the classifier options and trains the classifier.
template = templateTree(...
    'MaxNumSplits', 20);
classificationEnsemble = fitensemble(...
    predictors, ...
    response, ...
    'RUSBoost', ...
    100, ...
    template, ...
    'Type', 'Classification', ...
    'LearnRate', 0.1, ...
    'ClassNames', [0; 1],'nprint',10);

% Compact the classification ensembl
% Important for saving space!!
classificationEnsemble = compact(classificationEnsemble);

trainedClassifier.ClassificationEnsemble = classificationEnsemble;
convertMatrixToTableFcn = @(x) table(x, 'VariableNames', {'column'});
splitMatricesInTableFcn = @(t) [t(:,setdiff(t.Properties.VariableNames, {'column'})), array2table(table2array(t(:,{'column'})), 'VariableNames', predictorVar)];
extractPredictorsFromTableFcn = @(t) t(:, predictorNames);
predictorExtractionFcn = @(x) extractPredictorsFromTableFcn(splitMatricesInTableFcn(convertMatrixToTableFcn(x)));
ensemblePredictFcn = @(x) predict(classificationEnsemble, x);
trainedClassifier.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));
% Convert input to table
inputTable = table(trainingData);
inputTable.Properties.VariableNames = {'column'};

% Split matrices in the input table into vectors
inputTable = [inputTable(:,setdiff(inputTable.Properties.VariableNames, {'column'})), array2table(table2array(inputTable(:,{'column'})), 'VariableNames', allVars)];

% Extract predictors and response
% This code processes the data into the right shape for training the
% classifier.
predictorNames = predictorVar;
predictors = inputTable(:, predictorNames);
response = inputTable(:,responseVar);


% Perform cross-validation
if crossval
    partitionedModel = crossval(trainedClassifier.ClassificationEnsemble, 'KFold', 5);

    % Compute validation accuracy
    validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

    % Compute validation predictions and scores
    [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
else
    validationAccuracy = -1;
end
end