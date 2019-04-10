function [RI, VOI,SC] = evaluation_reg_image(inFile, gtFile, nthresh)
% function [thresh, cntR, sumR, cntP, sumP, cntR_best] = evaluation_reg_image(inFile, gtFile, evFile2, evFile3, evFile4, nthresh)
%
% Calculate region benchmarks for an image. Probabilistic Rand Index, Variation of
% Information and Segmentation Covering. 
%
% INPUT
%	inFile  : Can be one of the following:
%             - a collection of segmentations in a cell 'segs' stored in a mat file
%             - an ultrametric contour map in 'doubleSize' format, 'ucm2'
%               stored in a mat file with values in [0 1].
%
%	gtFile	:   File containing a cell of ground truth segmentations
%   evFile2, evFile3, evFile4  : Temporary output for this image.
%	nthresh:    Number of scales evaluated. If input is a cell of
%               'segs', nthresh is changed to the actual number of segmentations
%
%
% OUTPUT
%	thresh		Vector of threshold values.
%	cntR,sumR	Ratio gives recall.
%	cntP,sumP	Ratio gives precision.
%
%
% Pablo Arbelaez <arbelaez@eecs.berkeley.edu>


nthresh =1;

pre_label=load(inFile); 
segs = pre_label.label_matrix+1;
if exist('ucm2', 'var'),
    ucm = double(ucm2);
    clear ucm2;
elseif ~exist('segs', 'var')
    error('unexpected input in inFile');
end

gt_label=load(gtFile);
% disp(length(find(gt_label.label_matrix==0)));
groundTruth=gt_label.label_matrix+1;
nsegs = 1;

thresh = 1:nthresh; thresh=thresh';

regionsGT = [];
total_gt = 0;
for s = 1 : nsegs
    groundTruth = double(groundTruth);
    regionsTmp = regionprops(groundTruth, 'Area');
    regionsGT = [regionsGT; regionsTmp];
    total_gt = total_gt + max(groundTruth(:));
end

% zero all counts
cntR = zeros(size(thresh));
sumR = zeros(size(thresh));
cntP = zeros(size(thresh));
sumP = zeros(size(thresh));
sumRI = zeros(size(thresh));
sumVOI = zeros(size(thresh));

best_matchesGT = zeros(1, total_gt-1);

for t = 1 : nthresh,
    
    if exist('segs', 'var')
        seg = double(segs);
    else
        labels2 = bwlabel(ucm <= thresh(t));
        seg = labels2(2:2:end, 2:2:end);
    end
    
    [ri voi] = match_segmentations2(seg, groundTruth);
    sumRI(t) = ri;
    sumVOI(t) = voi;
    
    [matches] = match_segmentations(seg, groundTruth);
    matchesSeg = max(matches, [], 2);
    matchesGT = max(matches, [], 1);

    regionsSeg = regionprops(seg, 'Area');
    for r = 2 : numel(regionsSeg)
        cntP(t) = cntP(t) + regionsSeg(r).Area*matchesSeg(r-1);
        sumP(t) = sumP(t) + regionsSeg(r).Area;
    end
    
    for r = 2 : numel(regionsGT),
        cntR(t) = cntR(t) +  regionsGT(r).Area*matchesGT(r-1);
        sumR(t) = sumR(t) + regionsGT(r).Area;
    end
    
    best_matchesGT = max(best_matchesGT, matchesGT);

end

% output
cntR_best = 0;
for r = 2 : numel(regionsGT),
    cntR_best = cntR_best +  regionsGT(r).Area*best_matchesGT(r-1);
end

SC = cntR_best/(256*256-regionsGT(1).Area);
RI=sumRI;
VOI=sumVOI;



