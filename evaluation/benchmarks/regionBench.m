function [RI_vector,VOI_vector,SC_vector]=regionBench(imgDir, gtDir, inDir, nthresh)
% regionBench(imgDir, gtDir, inDir, outDir, nthresh)
%
% Run region benchmarks on dataset: Probabilistic Rand Index, Variation of
% Information and Segmentation Covering. 
%
% INPUT
%   imgDir: folder containing original images
%   gtDir:  folder containing ground truth data.
%   inDir:  folder containing segmentation results for all the images in imgDir. 
%           Format can be one of the following:
%             - a collection of segmentations in a cell 'segs' stored in a mat file
%             - an ultrametric contour map in 'doubleSize' format, 'ucm2' stored in a mat file with values in [0 1].
%   outDir: folder where evaluation results will be stored
%	nthresh	: Number of points in precision/recall curve.
%
% Pablo Arbelaez <arbelaez@eecs.berkeley.edu>




iids = dir(fullfile(imgDir,'*.png'));
if isempty(iids)
    iids = dir(fullfile(imgDir,'*.jpg'));
end
RI_vector=zeros(1,numel(iids));
VOI_vector=zeros(1,numel(iids));
SC_vector=zeros(1,numel(iids));
for i = 1 : numel(iids),
    inFile = fullfile(inDir, strcat(iids(i).name(1:end-4), '.mat'));
    gtFile = fullfile(gtDir, strcat(iids(i).name(1:end-4), '.mat'));
    
    [RI, VOI,SC]=evaluation_reg_image(inFile, gtFile);
    RI_vector(i)=RI;
    VOI_vector(i)=VOI;
    SC_vector(i)=SC;
end



