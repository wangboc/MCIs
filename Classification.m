% Classification for HC, MCI and AD
% HDU, Bocheng Wang 2018.10
tic;
clear;
addpath(genpath(pwd));
%% load data
load('./Data/BCTs/0.HC.mat');
Subject_HC = subjects;

load('./Data/BCTs/1.EMCI.mat');
Subject_EMCI = subjects;

load('./Data/BCTs/2.MCI.mat');
Subject_MCI = subjects;

load('./Data/BCTs/3.LMCI.mat');
Subject_LMCI = subjects;

load('./Data/BCTs/4.AD.mat');
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

 
%% Filter Feature selection
[FilteredMatrix, FilterdIndex] = Filter_Feature_Rank_importance(HC_vs_MCI, 1/2);
%% Wrapper Feature selection
Selected_train_data = WrapperFeatureSelection(FilteredMatrix);
toc;
