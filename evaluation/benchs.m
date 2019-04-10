%% Examples of benchmarks for different input formats
addpath benchmarks
clear all;close all;clc;

nthresh = 1;
ri=zeros(30,1);
voi=zeros(30,1);
sc=zeros(30,1);
category_lists=dir('pre_label/DB/');
score_dir = 'pre_label/score/';
if ~exist(score_dir,'file')
    mkdir(score_dir);
end
M_ri=0;
M_voi=0;
M_sc=0;
for category_idx=1:length(category_lists)
    category = category_lists(category_idx).name;
    if ~(strcmp(category,'.')||strcmp(category,'..'))
        imgDir = ['PG_data/new_image/',category];
        gtDir = ['PG_data/new_groundtruth/',category];
        inDir = ['pre_label/DB/',category];
        outDir = ['PG_data/evalue/DB/',category,'.mat'];

        iids = dir(fullfile(imgDir,'*.png'));
        if isempty(iids)
            iids = dir(fullfile(imgDir,'*.jpg'));
        end
        score_vector=zeros(3,numel(iids));
        best_score_class = cell(numel(iids),1);
        for i = 1 : numel(iids)
            inFile= [inDir,'/', strcat(iids(i).name(1:end-4)),'.mat'];
            gtFile = fullfile(gtDir, strcat(iids(i).name(1:end-4), '.mat'));

            

            [RI, VOI,SC]=evaluation_reg_image(inFile, gtFile, nthresh);
            score_vector(1,i)=RI;
            score_vector(2,i)=VOI;
            score_vector(3,i)=SC;
        score_vector
        save([score_dir,'/',strcat(iids(i).name(1:end-4)),'.mat'],'score_vector');
        end
    end
end
