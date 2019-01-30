function fNoisy = getEnsemble(f, densityModel, numMembers, noiseLevel)

[dim1, dim2] = size(f);
uncertainData = zeros(dim1,dim2,numMembers);

if(strcmp(densityModel,'uniform'))
    
for i=1:dim1
    for j=1:dim2
        uncertainData(i,j,:) = f(i,j) + uniRand(-noiseLevel,noiseLevel,numMembers);
    end
end

fNoisy = uncertainData;
    
end


