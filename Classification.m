% Classification for HC, MCI and AD
% HDU, Bocheng Wang 2018.10

tic;
clear all;
clc;
addpath(genpath(pwd));

%% Parameters 
% filterFS: {'NCA', 'Rank', 'None'}
% UsePresetSequence_in_WrapperFS: {1, 0}

filterFS = 'Rank';
UsePresetSequence_in_WrapperFS = 0;

%% load data
load('./Data_with_HC=24/BCTs/0.HC.mat');
Subject_HC = subjects;

load('./Data_with_HC=24/BCTs/1.EMCI.mat');
Subject_EMCI = subjects;

load('./Data_with_HC=24/BCTs/2.MCI.mat');
Subject_MCI = subjects;

load('./Data_with_HC=24/BCTs/3.LMCI.mat');
Subject_LMCI = subjects;

load('./Data_with_HC=24/BCTs/4.AD.mat');
Subject_AD = subjects;

HC_vs_EMCI   = cat(1, Subject_HC, Subject_EMCI);
HC_vs_MCI    = cat(1, Subject_HC, Subject_MCI);
HC_vs_LMCI   = cat(1, Subject_HC, Subject_LMCI);
HC_vs_AD     = cat(1, Subject_HC, Subject_AD);

EMCI_vs_MCI  = cat(1, Subject_EMCI, Subject_MCI);
EMCI_vs_LMCI = cat(1, Subject_EMCI, Subject_LMCI);
EMCI_vs_AD   = cat(1, Subject_EMCI, Subject_AD);

MCI_vs_LMCI  = cat(1, Subject_MCI, Subject_LMCI);
MCI_vs_AD    = cat(1, Subject_MCI, Subject_AD);

LMCI_vs_AD   = cat(1, Subject_LMCI, Subject_AD);

HC_vs_MCI_AD = cat(1, HC_vs_MCI, Subject_AD);
HC_vs_EMCI_vs_LMCI_vs_AD = cat(1, HC_vs_EMCI, LMCI_vs_AD);

% delete subgraph centrality
HC_vs_EMCI_vs_LMCI_vs_AD(:, 4336:4695) = [];
for index = 2:size(HC_vs_EMCI_vs_LMCI_vs_AD, 2)
    HC_vs_EMCI_vs_LMCI_vs_AD(:, index) = mapminmax(HC_vs_EMCI_vs_LMCI_vs_AD(:, index)')';
end

%% Filter Feature selection
if strcmp(filterFS, 'Rank')
    [FilteredMatrix, FilterdIndex] = Filter_Feature_Rank_importance(HC_vs_EMCI_vs_LMCI_vs_AD, 1/3);
elseif strcmp(filterFS, 'NCA')
    [FilteredMatrix, FilterdIndex] = NCA(HC_vs_EMCI_vs_LMCI_vs_AD);
elseif strcmp(filterFS, 'Predefined')    
    X = HC_vs_EMCI_vs_LMCI_vs_AD(:, 2:size(HC_vs_EMCI_vs_LMCI_vs_AD, 2));
    y = HC_vs_EMCI_vs_LMCI_vs_AD(:, 1);
    FilterdIndex = [2281 2284 3155 2205 1331 1079 1083 1688 1652 1658 1800 2155 1120 1691 ...
        1901 2642 2834 1135 1379 1564 838 2567 713 1120 274 832 2178 2876 2312 ...
        962 812 2884 517 1198 2608 794 3213 846 739 1351 1979 919 872 2120 784 ...
        1346 909 409 1067 523 2503 2178 2323 2145 1074 2399]; 
       
    FilteredMatrix = [y X(:, FilterdIndex)];
end
%% Wrapper Feature selection
[Selected_train_data, SelectedFeatures_in_RankImportanceOrder] ...
    = WrapperFeatureSelection(FilteredMatrix, UsePresetSequence_in_WrapperFS);
% if strcmp(filterFS, 'Predefined') ~= true
%     RankImportanceOrder_2_FeatureName(FilterdIndex, SelectedFeatures_in_RankImportanceOrder);
% end
%% Matlab Machine learning Toolbox ...
%% libSVM tools
libSVM_result_filename = 'tempfiles\libSVM_result.txt';
matrix2libsvmformat(Selected_train_data, libSVM_result_filename);
libSVM_Accuracy_Output = evaluateSVM(libSVM_result_filename);
toc;
