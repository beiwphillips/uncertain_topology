[X, Y, gt] = generateSynthetic();

noisyEnsemble = getEnsemble(gt, 'uniform', 50);

%subLevelSetUncertainTopology(gt,noisyEnsemble);
%[rightP, upP, leftP, downP] = probabilisticMarchingGradient(X, Y, gt, noisyEnsemble);

%mandatory = findMandatoryMaxima(noisyEnsemble);

%[minProb, maxProb, saddleProb, regularProb] = fourNeighborsComputeAndVisMinimaProbability(noisyEnsemble);

% figure
% imagesc(gt)
% figure
% imagesc(maxProb);
% colorbar
% figure

%displayContourTree(gt);

%displayContourTree(noisyEnsemble(:,:,1));

%displayContourTree(mean(noisyEnsemble,3));

