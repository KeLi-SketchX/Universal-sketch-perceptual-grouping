clear
data_root_path = '/import/vision-datasets001/kl303/PG_data/annotations/';
category_list = dir(data_root_path);
for category_idx = 1:length(category_list)
    category = category_list(category_idx).name;
    if ~(strcmp(category,'.')||strcmp(category,'..'))
        sketch_list = dir([data_root_path,category,'/label/*.txt']);
        sketch_ids = zeros(length(sketch_list),1);
        for id =1:length(sketch_list)
            sketch_name = str2double(sketch_list(id).name(1:end-4));
            sketch_ids(id) = sketch_name;
        end
        sketch_ids = sort(sketch_ids,'ascend');
        txt_name = ['./PG_data/test_file/',category,'.txt'];
        fid =fopen(txt_name,'w');
        for i=701:length(sketch_ids)
%        for i=1:650
            id = int2str(sketch_ids(i));
            str_id = my_zfill(id,4);
            fprintf(fid,'%s.txt\n',str_id);
        end
        fclose(fid);
        
    end
end
