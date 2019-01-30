foo = "ackley";
noiseLevel = 0.8;

[X, Y, gt] = generateSynthetic(foo);

noisyEnsemble = getEnsemble(gt, 'uniform', 50, noiseLevel);

save(strcat("../data/", foo, "_groundtruth.mat"), "gt", '-V7')
save(strcat("../data/", foo, "_", num2str(noiseLevel) , "_uncertain.mat"), "noisyEnsemble", '-V7')

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

