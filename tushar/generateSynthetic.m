% Generate synthetic data

function [X, Y, gt] = generateSynthetic()

rng('default')

%% Function 1
% x = -4 : 0.1: 4;
% y = -4: 0.1: 4;
% /home/sci/tushar.athawale/myResearch/myAcceptedPapers/tusharProjects/vis18/revisedVis18/src/MATLAB/compactFinalKdeCode
% f = zeros(numel(x), numel(y));
% 
% for i=1:numel(x)
%     for j=1:numel(y)
%         pos = [x(i),y(j)];
%         f(i,j) = norm([-1,0] - pos) * norm([1,0] - pos) ;        
%     end
% end
% 
% f = f*10;

%% Function 2
[X, Y] = meshgrid(0:0.025:1,0:0.025:1);

f =   0.5 * exp(-(X - 0.25).^2./0.3^2)...
    + 0.5 * exp(-(Y - 0.25).^2./0.3^2)...
    + 0.5 * exp(-(X - 0.75).^2./0.1^2)...
    + 0.5 * exp(-(Y - 0.75).^2./0.1^2);

f = f*100;

gt = f;

% Function 3
% [X, Y] = meshgrid(0:0.2:1,0:0.2:1);
% 
% f =   0.5 * exp(-(X - 0.25).^2./0.3^2)...
%     + 0.5 * exp(-(Y - 0.25).^2./0.3^2)...
%     + 0.5 * exp(-(X - 0.75).^2./0.1^2)...
%     + 0.5 * exp(-(Y - 0.75).^2./0.1^2);
% 
% f = f*100;

gt = f;

% Add noise to create ensemble data

% numEnsemble = 10;
% 
% uncertainData = zeros(numel(x),numel(y),numEnsemble);
% 
% for i=1:numel(x)
%     for j=1:numel(y)
%         uncertainData(i,j,:) = f(i,j) + uniRand(-0.2,0.2,numEnsemble);
%     end
% end
% 
% gt = f;
% noisyEnsemble = uncertainData;

% % Visualize Jacobi level set
% 
% [d1x,d1y] = gradient(uncertainData(:,:,1));
% % contour(x,y,f);
% % hold on
% % quiver(x,y,d1x,d1y)
% % hold off
% 
% [d2x,d2y] = gradient(uncertainData(:,:,2));
% 
% % contour(x,y,f);
% % hold on
% % quiver(x,y,d2x,d2y)
% % hold off
% 
% % Compute comparison measure
% k = zeros(numel(x), numel(y));
% for i=1:numel(x)
%     for j=1:numel(y)
%         k(i,j) = norm(cross([d1x(i), d1y(j), 0], [d2x(i), d2y(j), 0]));
%     end
% end
% 
% % Extract zero level set from comparison measure
% isocontour(k,0.01);
% hold on
% isocontour(uncertainData(:,:,1), 1);


