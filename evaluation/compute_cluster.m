clear
category_list = dir('./PG_data/groundtruth/');
for category_idx =1:length(category_list)
    category = category_list(category_idx).name;
    if ~(strcmp(category,'.')||strcmp(category,'..'))
        data_list=dir(['./PG_data/groundtruth/',category,'/*.mat']);
        category_cluster_num=zeros(length(data_list),1);
        for data_idx = 1:length(data_list)
            data_name = data_list(data_idx).name;
            load(['./PG_data/groundtruth/',category,'/',data_name]);
            category_cluster_num(data_idx,1)=length(unique(label_matrix))-1;
        end
        out_put_str = ['category: ',category,' min: ',int2str(min(category_cluster_num)),' max: ',int2str(max(category_cluster_num)),' average: ',num2str(sum(category_cluster_num)/length(data_list))];
        disp(out_put_str);
    end
end