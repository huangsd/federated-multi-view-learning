clear
load('sonar.mat')
lambda = [0.1:0.2:2]; 
mu = [1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3];
rho = [1e-1 1e0 1e1 1e2 1e3];
varrho = [1e-1 1e0 1e1 1e2 1e3];

a = 0;
b = 0;
i = 1;
while a < 1e2
    a = 0.1*(2^b);
    ab(i) = a;
    i = i + 1;
    b = b + 0.5;
end

param = [];
% param.bottom = 0.9;
j = 2;
param.maxIter = 35;
param.is_alpha = 1;
param.rho = ab(j);
param.varrho = ab(j);
param.mu = mu(9);
param.lambda = lambda(3);

% param.bottom = 0;
% [G, VV, VVres, Gres, Obj] = FedMVL(data, labels, param);

ii = [1 0.9 0.75 0.5];
for i = 1:4
    param.bottom = ii(i)
    [G, VV, VVres, Gres, Obj] = FedMVL(data, labels, param);
    OObj(i,:) = Obj;
end
plot(OObj')

% for j = 1:10
%     j
%     param.lambda = lambda(j);
%    [G, VV, VVres, Gres, Obj] = FedMVL(data, labels, param);
% end


% t = 1;
% for k = 1:length(lambda)
%     for i = 1:length(ab)
%         for j = 1:length(ab)
%             param.lambda = lambda(k);
%             param.rho = ab(i); % ab(5)
%             param.varrho = ab(j); % ab(12)
%             [G, VV, VVres, Gres, Obj] = FedMVL(data, labels, param);
%             Vress(t,:) = VVres
%             Gress(t,:) = Gres
%             t = t +1;
%         end
%     end
% end

% 
% param.rho = ab(5); % ab(5)
% param.varrho = ab(12); % ab(12)
% [G, VV, VVres, Gres, Obj] = FedMVL(data, labels, param);

% k-means
% dataall = [data{1}',data{2}'];
% for i = 1:10
%    res = kmeans(dataall, length(unique(labels)), 'emptyaction', 'singleton');
%    Gres(i,:) = ClusteringMeasure(labels, res);
% end
% mean(Gres)
% std(Gres)

