function [SelectedTrainData] = WrapperFeatureSelection(Matrix)
%% Implement sequentialfs Matlab function
% HDU, Bocheng Wang 2018.10
%%
X = Matrix(:, 2:size(Matrix, 2));
y = Matrix(:, 1);
X = mapstd(X')';

c = cvpartition(y,'KFold',10);
opts = statset('display', 'iter',  'TolTypeFun','rel', 'UseParallel', true);
fun = @(train_data,train_labels,test_data,test_labels) ...
       sum(predict(fitcsvm(train_data,train_labels,'KernelFunction','rbf'), test_data) ~= test_labels); 

[fs,history] = sequentialfs(fun,X,y,'cv',c,'options',opts);
SelectedLabel = 1:size(Matrix, 2);
SelectedLabel = SelectedLabel(fs);
SelectedTrainData = [X(:, SelectedLabel), y];
%% temp results 
% HC_vs_EMCI 
% [ 2 73];                           Accuracy:95.8%
% [ 2 3 14];                         Accuracy:97.9%
%% temp results 
% HC_vs_MCI
% [ 9    41];                        Accuracy:97.9%
% [1 228];                           Accuracy:95.8%
%% temp results 
% HC_vs_LMCI
%% temp results 
% HC_vs_AD
%% temp results 
% EMCI_vs_MCI
%% temp results 
% EMCI_vs_LMCI
% [456] Accuracy: 77.1%
%% temp results 
% EMCI_vs_AD
% [25 177 905 935]; Accuracy:85.4%
%% temp results 
% MCI_vs_LMCI
% [2 10 24 ]; Accuracy:75%
%% temp results 
% MCI_vs_AD
% [70 136 365 606 1585]; Accuracy:83.3%
%% temp results 
% LMCI_vs_AD
% [24 28 254 1514]; Accuracy:87.5%


