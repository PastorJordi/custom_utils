function [RT,v,Accu,bound,s] = D2M_simulation6(t_c,c,b,d,v_u,a_u,t_0_u,v,a_e,z_e,t_0_e,StimOnset,prop,dt,n,x_cont_inf,post_correct)
%D2M simulations under parameters
%"v_u","a_u","t_0_u","v_e","a_e","z_e","t_0_e". The numerical error allowed
%for the computation of pdfs is "err", and the time-sep and the number of
%realizations for the simulation of the urgency integrator's accuracy are
%respectively "dt" and "n".
RT = nan(1,length(n));
Accu = RT;
bound = zeros(1,length(n));
%% Block structure (for stimulus-coding of transition bias in starting offset z_e)
prep_rep = .7;
prep_alt = .2;
Block_vec = mod(ceil((1:n)/200),2);
Rep_vec = Block_vec.*(2*binornd(1,prep_rep,[1 n])-1)+~Block_vec.*(2*binornd(1,prep_alt,[1 n])-1);
Block_vec = 2*Block_vec-1;
%% Markovian stimulus
Categ_vec=zeros(size(Block_vec)); Categ_vec(1)=2*binornd(1,.5)-1;
for i=2:n
    Categ_vec(i) = Categ_vec(i-1)*Rep_vec(i);
end
if post_correct~=1
    post_correct = 0; %for later sing in the starting offset.
end
%% Difficulty
if length(v)>1
    v(v==min(v))=0; %to force zero coherence as random stimulus.
end
Ll = length(v);
Dif=ceil(Ll*rand([1 n])); %random number between 1 and 4 for difficulty at each trial
%% Response category
Resp_vec=zeros(size(Block_vec)); Resp_vec(1)=2*binornd(1,.5)-1;
%% Modulation with trial index
rr = rand(1,n-1);
ss = rand(1,n-1);
uu = rand(1,n-1);
aa = rand(1,n-1);
%% D2M
mu = (z_e+1)/2; % transformed z_e for mean of beta distribution of starting offset
s = 0.2;%.2; % std of the beta distribution
a = (1-mu)*(mu/s)^2-mu; % parameter a for the beta distribution
B = a*(1/mu-1); % parameter b for the beta distribution
if z_e==0
    zz=0;
else
    zz=1;
end
tic
for i=2:n
    vv_u = v_u + prop.v_trial*prop.trial(2,sum(prop.trial(1,:)<rr(i-1))+1);
    vv_u = vv_u + prop.v_rewsize*prop.rewardsize(2,sum(prop.rewardsize(1,:)<ss(i-1))+1);
    vv_u = vv_u + prop.v_cumrew*prop.cumreward(2,sum(prop.cumreward(1,:)<uu(i-1))+1);
    l = Dif(i);
    v_e = v(l)*Categ_vec(i);
    x_u = 0; % initialize Urgency Variable
    x_e = zz*(2*betarnd(a,B)-1)*a_e; % initialize Decision Variable with std so that performance is around 60%
    x_e = (~post_correct*Resp_vec(i-1) + post_correct*Categ_vec(i-1))*Block_vec(i)*x_e; %aftercorrect fit, so that it is really categ
    k = floor(t_0_u/dt); % initialize DDM iteration
    while (x_u<a_u)&&(k*dt<t_0_e) % for FBs and Express Responses (purely urgency-triggered responses)
        x_u = x_u + vv_u*dt + sqrt(dt)*randn; % urgency integrator
        k = k+1;
    end
    while (x_u<a_u)&&(abs(x_e)<a_e) % for Non-Express Responses
        x_u = x_u + vv_u*dt + sqrt(dt)*randn; % urgency integrator
        x_e = x_e + v_e*dt + sqrt(dt)*randn; % evidence integrator
        k = k+1; % time step counter
    end
    RT(i) = k*dt; % the RT is already set, but the choice is not set yet for urgency-triggered trials
    bound(i) = x_u<a_u; % whether the trial hit the evidence bound or the urgency bound
    if ~bound(i)&&StimOnset<RT(i) % for urgency-triggered trials, integrate evidence for "(t_0_e-StimOnset)" seconds more
        T = RT(i); % we save the hitting time
        if T<t_0_e % for express responses
            k = floor(t_0_e/dt); % we cannot integrate evidence until "t_0_e"
        end
        while k*dt<T+(t_0_e-StimOnset) % keep integrating for "(t_0_e-StimOnset)" seconds more
            x_e = x_e + v_e*dt + sqrt(dt)*randn; % evidence integrator
            k = k+1; % time step counter
        end
    end
    Resp_vec(i) = sign(x_e);
    Accu(i) = (Resp_vec(i)*Categ_vec(i)+1)/2;
end
disp(['Time for simulations', num2str(toc)])
bound = logical(bound);
%% Contribution of contaminants as chance responses (if we had "n" no contaminants, them we have a mean of n*c/(1-c))
N.noncont = n;
N.cont.all = nbinrnd(n,1-c); %negative binomial for number of failures (contaminants) before "n" successes.
N.cont.exp = binornd(N.cont.all,d); %number of exponential contaminants.
N.cont.uni = N.cont.all - N.cont.exp; % number of uniform contaminants.
pd.exp = makedist('Exponential','mu',1/b); %exponential contaminant distro.
pd.exp = truncate(pd.exp,0,t_c); %truncated exponential
pd.uni = makedist('Uniform','Lower',0,'Upper',t_c); %uniform contaminant distro.
RT_cont = [random(pd.exp,1,N.cont.exp) random(pd.uni,1,N.cont.uni)];
RT = [RT RT_cont];
bound = [bound nan(1,N.cont.all)];
Dif_cont = ceil(Ll*rand([1 N.cont.all]));
Dif = [Dif Dif_cont];
Accu_cont = binornd(1,.5,1,N.cont.all); % start making all contaminants random.
cont_inf = logical(binornd(1,x_cont_inf,1,N.cont.all)); % A random fraction "x_cont_inf" of contaminants to use unbounded EA.
Categ_vec_idx_cont = rem(find(cont_inf),length(Categ_vec)); Categ_vec_idx_cont(Categ_vec_idx_cont==0) = length(Categ_vec);
Categ_vec_cont = Categ_vec(Categ_vec_idx_cont); % Categ_vec for these contaminants.
V_cont_inf = v(Dif_cont(cont_inf)).*Categ_vec_cont; % EA drift for these contaminants.
Evid_t = (abs(RT_cont(cont_inf)-StimOnset)+(RT_cont(cont_inf)-StimOnset))/2; % EA integration time vs RT (rectified linear).
Mu = z_e*a_e+V_cont_inf.*Evid_t; Sigma = sqrt(s.^2+Evid_t); % Mean and sigma of unbounded EA vs RT.
Accu_cont(cont_inf) = binornd(1,normcdf(0,Mu.*Categ_vec_cont,Sigma,'upper'),1,sum(cont_inf));
Accu = [Accu Accu_cont];
%% for latter plotting
v = v(Dif); %for later plotting at rat_reactiontimes
end