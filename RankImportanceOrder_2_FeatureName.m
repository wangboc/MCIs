function  RankImportanceOrder_2_FeatureName(FilterdIndex, SelectedFeatures_in_RankImportanceOrder)

% for test   SelectedFeatures_in_RankImportanceOrder = [10 234 396 546 692 ];

features_index = FilterdIndex(SelectedFeatures_in_RankImportanceOrder);
for index = 1:size(features_index, 2)
    feature = features_index(index);
    feature_name_index = fix(feature/360);    
    feature_location = rem(feature, 360);
    [name, area] = ParseFeature(feature_name_index, feature_location);
    disp(['分区ID：' num2str(feature_location) ' 特征类型：' name '  特征所在分区：' char(area)]);
end

