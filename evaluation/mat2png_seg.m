clear
clear
label_root_path ='/import/vision-datasets/kl303/PycharmProjects/BSR/bench/pre_label/segmentatiom/';
data_root_path = '/import/vision-datasets001/kl303/PG_data/annotations/';
category_lists = dir(label_root_path);
for category_idx = 1:length(category_lists)
    category = category_lists(category_idx).name;
    if ~(strcmp(category,'.')||strcmp(category,'..'))
        label_file_lists = dir([label_root_path,category,'/*.mat']);
        for label_file_idx =1:length(label_file_lists)
            label_file_name = label_file_lists(label_file_idx).name;
            pre_label = load([label_root_path,category,'/',label_file_name]);
            data_file_path = [data_root_path,category,'/data/',label_file_name];
            data = load(data_file_path);
            sketch = data.new_sketch(:,:,1);
            unique_sketch = unique(sketch);
            label_matrix = sketch;
            bg_idx = find(sketch==255);
            label_matrix(bg_idx)=0;
            for unique_idx = 1:length(unique_sketch)
                unique_value = unique_sketch(unique_idx);
                if unique_value~=255
                    group_label = pre_label.label(unique_value)+1;
                    label_idx = find(sketch==unique_value);
                    label_matrix(label_idx)=group_label;
                end
            end
            evaluate_file_path = ['/import/vision-datasets/kl303/PycharmProjects/BSR/bench/PG_data/segmentatiom/',category,'/',label_file_name];
            if ~exist(['/import/vision-datasets/kl303/PycharmProjects/BSR/bench/PG_data/segmentatiom/',category],'file')
                mkdir(['/import/vision-datasets/kl303/PycharmProjects/BSR/bench/PG_data/segmentatiom/',category]);
            end
            save(evaluate_file_path,'label_matrix');
        end
    end
end

