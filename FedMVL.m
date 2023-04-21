%
function [G, VV, VVres, Gres, Obj] = FedMVL(inXCell, labels, param)
%
% Federated multi-view learning (FedMVL) -- for Clustering
%
% ATTN1: This package is free for academic usage. The code was developed by Mr. Shudong Huang (huangsd@std.uestc.edu.cn). 
% You can run it at your own risk. For other purposes, please contact Prof. Zenglin Xu (zlxu@uestc.edu.cn)
%
% solve the following problem
% min_{U^(m), V^(m), V, G, alpha^{m}} sum_m {(alpha^{m})^r*||X^(m) -
% U^(m)V(m)^T||_{F}^{2} + mu/2||G^{T}V - I||_{F}^{2}
% s.t. sum_m{alpha^{m}) = 1, alpha^{m} >= 0, V^(m) >= 0, V = V^(m), G = V
% 
% input: 
%       inXcell: M by 1 cell, and the size of each cell is d_m by n
%       Cnum: number of clusters
%       lambda: hyperparameter to control the distribution of the weights, 
%               which is searched in logarithm form 0.1 to 2 with step size 0.2
%       inPara: parameter cell
%               inPara.maxIter: max number of iterator
%               inPara.thresh:  the convergence threshold
%               inPara.numCluster: the number of clusters
%               inPara.r: the parameter to control the distribution of the weights for each view
%               inPara.sys_het = 1; % run systems (1) or stats heterogeneity exps (0)
%               inPara.top = 1.0; % highest number of rounds
%               inPara.bottom = 0.1; % lowest number of rounds
%       inG0: init common cluster indicator
%       is_alpha: alpha parameter-weighted (1) or adaptively-weighted (0)
%
% output:
%       outG: the global common matrix G (n by c)
%       outV: the global common matrix V (n by c) 
%       outUcell: the cluster centroid for each view (d_m by c by M)
%       outObj: obj value
%       outNumIter: number of iterator
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ref:
% Shudong Huang, Wei Shi, Zenglin Xu, Ivor Tsang 
% Iterative Orthogonal Federated Multi-view Learning
% The 1st International Workshop on Federated Machine Learning for User Privacy and Data Confidentiality (FML'19)
% in conjunction with the 28th International Joint Conference on Artificial Intelligence (IJCAI-19).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ATTN2: This package was developed by Mr. Shudong Huang (huangsd@std.uestc.edu.cn). For any problem concerning the code, 
% please feel free to contact Mr. Huang.
%
% e.g.,
inPara = [];
inPara.thresh = 1e-6;
Cnum = length(unique(labels));
inPara.numCluster = Cnum;
% r: logarithm from 0.1 to 2 with step size 0.2
lambda = param.lambda;
is_alpha = param.is_alpha;
inPara.r = 10^lambda; 
inPara.sys_het = 1; % run systems (1) or stats heterogeneity exps (0)
inPara.top = 1; 
inPara.bottom = param.bottom; % low variability environments (0.9) or high variability (0.1)
%
% tic;
%
%% # of views 
nView = length(inXCell);
%% # of samples
nSamp = size(inXCell{1}, 2);
%% # of features
nFea = zeros(nView,1);
for m = 1:nView
    nFea(m) = size(inXCell{m}, 1);
end
%% parameter settings
maxIter = param.maxIter;
thresh = inPara.thresh;
r = inPara.r;
% aggregation parameter
gamma = 1;
%
sigma = nView; % nView

rho = param.rho; % = 
varrho = param.varrho; % = [1 10]
mu = param.mu; % 1e-4

tmp = 1/(1-r);
% Obj = zeros(maxIter, 1);
%% normalization 
% for m = 1:nView
%     inXCell{m}=mapminmax(inXCell{m},0,1);
% end
%
for m = 1:nView
    for  i = 1:nSamp
        inXCell{m}(:,i) = (inXCell{m}(:,i) - mean(inXCell{m}(:,i)))/std(inXCell{m}(:,i));
    end
end
%% initialize indicator matrix V^(m), G, and VV (i.e., V)
V = cell(nView,1);
VV = rand(nSamp,Cnum);
G = rand(nSamp,Cnum);
for m = 1:nView
    V{m} = rand(nSamp,Cnum);
end
%
% for m = 1:nView
%     initV = zeros(nSamp,Cnum);
%     initV0 = kmeans(inXCell{m}', Cnum, 'emptyaction', 'singleton');
%     for i = nSamp
%         initV(i,initV0(i)) = 1;
%     end
%     clear initV0
%     V{m} = initV+0.2;
% end
%% initialization
alpha = ones(nView,1)/nView; 
Psi = zeros(size(G));
Phi = cell(nView,1);
for m = 1:nView
    Phi{m} = zeros(size(V{m}));
end
%
%%
%
%% update
% loop
for iter = 1:maxIter
    %
    % 
    if mod(iter, 10) == 0
       % fprintf('numOfOutliers = %d, ratio = %f\n', length(Idx),ratio);
       % fprintf('%dth iteration, obj = %f \n', it, obj);
       fprintf('processing iteration %d...\n', iter);
    end
    %
    % loop over views (in parallel)  
    %
    % update U^{m}
    if iter == 1
        U = cell(nView,1);
    end
    for m = 1:nView       
        U1 = inXCell{m}*V{m};
        U2 = V{m}'*V{m};
        U{m} = U1/U2;
        clear U1 U2
    end
    %   
    % update V^{m}
    if (inPara.sys_het)
        sysnum = (inPara.top - inPara.bottom) .* rand(nView,1) + inPara.bottom;
    end
    for m = 1:nView
        Samp_idx = randperm(nSamp);
        % run systems heterogeneity or not  
        % randomly dropped samples of the nodes
        if (inPara.sys_het)
            sys_num = nSamp*(sysnum(m)); % ceil()
        else
            sys_num = nSamp;
        end      
        deltaV = zeros(size(V{m}));
        V1 = U{m}'*U{m};
        V2 = sigma*V1 + rho*eye(size(V1));
        clear V1
        V3 = (alpha(m)^r)*(inXCell{m} - U{m}*V{m}')';
        for i = 1:sys_num
            idx = Samp_idx(mod(i, nSamp) + 1);
            V4 = rho*(VV(idx,:) - V{m}(idx,:)) - Phi{m}(idx,:) + 2*V3(idx,:)*U{m};
            deltaV(idx,:) = V4/V2;
            clear V4
        end
        clear V2 V3
        % without updating deltaV or not 
        if (inPara.bottom == 0)
            gamma = 0;
        end
        V{m} =  V{m} + gamma*deltaV;
        clear deltaV
        % V{m}(V{m}<0) = 0;
        V{m} = max(V{m},0);
    end
    %
    % update V
    VV1 = mu*(G*G') + (varrho+rho*nView)*eye(nSamp);
    VV2 = Phi{1};
    VV3 = V{1};
    for m = 2:nView
        VV2 = VV2 + Phi{m};
        VV3 = VV3 + V{m}; 
    end
    VV3 = rho*VV3;
    VV4 = (mu + varrho)*G - Psi + VV2 + VV3;
    VV = VV1\VV4;
    clear VV1 VV2 VV3 VV4
    %
    % update G
    % G = VV + Psi/varrho;
    G1 = mu*(VV*VV') + varrho*eye(nSamp);
    G2 = (mu + varrho)*VV + Psi;
    G = G1\G2;
    clear G1 G2
    %
    % update \Psi
    Psi = Psi + varrho*(VV - G);
    %
    % update \Phi^{m}
    for m = 1:nView
        Phi{m} = Phi{m} + rho*( V{m} - VV);
    end
    %
    % update alpha
    Q = zeros(nView, 1);
    for m = 1:nView
        Q(m) = sum(sum((inXCell{m} - U{m}*V{m}').^2));
    end
    if (is_alpha)
        alpha = ((r*Q).^tmp)/(sum(((r*Q).^tmp)));
    else
        for m = 1:nView
            alpha(m) = 0.5/sqrt(sum(sum((inXCell{m} - U{m}*V{m}').^2)));
        end       
    end
    %
    % compute average residue
    if (iter > 0) % after Phi and Psi are updated
        obj = 0;
        for m = 1:nView
            obj1 = sum(sum((inXCell{m} - U{m}*V{m}').^2));
            obj = obj + obj1;
        end
        obj2 = 0.5*mu*sum(sum((G'*VV-eye(Cnum)).^2));
        obj = (obj+obj2)/nView;
        % obj = obj/nView + obj2;
        %obj = obj/nView;
        Obj(iter) = obj;
    end
    %
    % convergence or not
    if(iter > 1)
        diff = abs(Obj(iter-1) - Obj(iter));
        if(diff < thresh)
            break;
        end
    end
    %
 
end
%%
% debug
% figure
% plot(1: length(obj), obj);
% Tim=toc;
%
    res1 = kmeans(VV, length(unique(labels)), 'emptyaction', 'singleton');
    VVres = ClusteringMeasure(labels, res1)
    %
    res2 = kmeans(G, length(unique(labels)), 'emptyaction', 'singleton');
    Gres = ClusteringMeasure(labels, res2)
    %
    plot(Obj)
    %

end
%