gt = generateSynthetic();
[a,b] = size(gt);
[X,Y] = meshgrid(1:1:a, 1:1:b);
TRI = delaunay(X,Y);
vtktrisurf(TRI,X,Y,gt,'hills','testHills.vtu')
noisyEnsemble = getEnsemble(gt, 'uniform', 50);

vtktrisurf(TRI,X,Y,noisyEnsemble(:,:,1),'hillsNoise1','testHillsNoise1.vtu')
vtktrisurf(TRI,X,Y,noisyEnsemble(:,:,2),'hillsNoise2','testHillsNoise2.vtu')



