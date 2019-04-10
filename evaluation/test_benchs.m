%% Examples of benchmarks for different input formats
addpath benchmarks
clear all;close all;clc;


%% 1. all the benchmarks for results stored in 'ucm2' format

% imgDir = 'data/images';
% gtDir = 'data/groundTruth';
% inDir = 'data/ucm2';
% outDir = 'data/test_1';
% mkdir(outDir);
% nthresh = 5;
% 
% tic;
% allBench(imgDir, gtDir, inDir, outDir, nthresh);
% toc;
% 
% plot_eval(outDir);
% 
% %% 2. boundary benchmark for results stored as contour images
% 
% imgDir = 'data/images';
% gtDir = 'data/groundTruth';
% pbDir = 'data/png';
% outDir = 'data/test_2';
% mkdir(outDir);
% nthresh = 5;
% 
% tic;
% boundaryBench(imgDir, gtDir, pbDir, outDir, nthresh);
% toc;
% 
% %% 3. boundary benchmark for results stored as a cell of segmentations
% 
% imgDir = 'data/images';
% gtDir = 'data/groundTruth';
% pbDir = 'data/segs';
% outDir = 'data/test_3';
% mkdir(outDir);
% 
% nthresh = 99; % note: the code changes this to the actual number of segmentations
% tic;
% boundaryBench(imgDir, gtDir, pbDir, outDir, nthresh);
% toc;
% 
% 
% %% 4. all the benchmarks for results stored as a cell of segmentations
% 
% imgDir = 'data/images';
% gtDir = 'data/groundTruth';
% inDir = 'data/segs';
% outDir = 'data/test_4';
% mkdir(outDir);
% nthresh = 5;
% 
% tic;
% allBench(imgDir, gtDir, inDir, outDir, nthresh);
% toc;


%% region benchmarks for results stored as a cell of segmentations




%%SC

nthresh = 1;
ri=zeros(30,1);
voi=zeros(30,1);
sc=zeros(30,1);
category_lists=dir('pre_label/500unlabeled/');
% cost_thr = 'DB_label';
% disp(['current cost thr: ',cost_thr]);
for category_idx=1:length(category_lists)
    category = category_lists(category_idx).name;
%     category = 'airplane';
    if ~(strcmp(category,'.')||strcmp(category,'..'))
        imgDir = ['PG_data/new_image/',category];
        gtDir = ['PG_data/new_groundtruth/',category];
        inDir = ['pre_label/500unlabeled/',category];
%         outDir = ['PG_data/evalue/cluster/',category,'.mat'];
%         inDir = ['../../deeplab/pre_label/',category];
%         outDir = ['PG_data/evalue/trip/',category,'.mat'];
        [RI_vector,VOI_vector,SC_vector]=regionBench(imgDir, gtDir, inDir, nthresh);
        out_put_infor =struct;
        out_put_infor.RI_vector=RI_vector;
        out_put_infor.VOI_vector=VOI_vector;
        out_put_infor.SC_vector=SC_vector;
        out_put_infor.RI_acc=sum(RI_vector)/length(RI_vector);
        out_put_infor.VOI_acc=sum(VOI_vector)/length(VOI_vector);
        out_put_infor.SC_acc=sum(SC_vector)/length(SC_vector);
%         save(outDir,'-struct','out_put_infor');
        ri(category_idx,1)=out_put_infor.RI_acc;
        voi(category_idx,1)=out_put_infor.VOI_acc;
        sc(category_idx,1)=out_put_infor.SC_acc;
    end 
end

% save(['./ri_',cost_thr,'.mat'],'ri');
% save(['./voi_',cost_thr,'.mat'],'voi');
% save(['./sc_',cost_thr,'.mat'],'sc');


