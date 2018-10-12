function  RankImportanceOrder_2_FeatureName(FilterdIndex, SelectedFeatures_in_RankImportanceOrder)

% for test SelectedFeatures_in_RankImportanceOrder = [4 9 23 33 87 175 176 1054 ];

features_index = FilterdIndex(SelectedFeatures_in_RankImportanceOrder);
for index = 1:size(features_index, 2)
    feature = features_index(index);
    feature_name_index = fix(feature/360);    
    feature_location = rem(feature, 360);
    [name, area] = ParseFeature(feature_name_index, feature_location);
    disp(['�������ͣ�' name '  �������ڷ�����' char(area)]);
end

