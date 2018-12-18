% Generate synthetic data

function gt = generateSynthetic()

rng('default')

%% Function 1
% x = -4 : 0.1: 4;
% y = -4: 0.1: 4;
% 
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
[X, Y] = meshgrid(0:0.0025:1,0:0.0025:1);

f =   0.5 * exp(-(X - 0.25).^2./0.1^2)...
    + 0.5 * exp(-(Y - 0.25).^2./0.1^2)...
    + 0.5 * exp(-(X - 0.75).^2./0.1^2)...
    + 0.5 * exp(-(Y - 0.75).^2./0.1^2);

f = f*50;

gt = f;

