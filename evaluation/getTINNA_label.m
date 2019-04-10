clear
root_input='../../deeplab/TINNA_data/label/';
root_output='./PG_data/TINNA/groundtruth/';

category_lists = dir(root_input);

for category_idx = 1:length(category_lists)
    category_name = category_lists(category_idx).name;
    if ~(strcmp(category_name,'.')&&strcmp(category_name,'..'))
        if ~(exist([root_output,category_name],'dir'))
            mkdir([root_output,category_name])
            sketch_lists = dir([root_input,category_name,'/*.png']);
            for sketch_idx =1:length(sketch_lists)
                sketch_name=sketch_lists(sketch_idx).name;
                sketch_path=[root_input,category_name,'/',sketch_name];
                sketch = imread(sketch_path);
                label_path=[root_output,category_name,'/',sketch_name(1:end-4),'.mat'];
                save(label_path,'sketch');
            end
        end
    end
end