function warpshell_matlab_check()
% Independent MATLAB + WarpFactory reproduction of the v18 results, run on the repo's data.
%   Part 1: full boxed C_LP cross-solve via MATLAB linprog (reads certificate.mat, exported
%           from the repo's frozen matrices) -- cross-toolchain check vs the HiGHS value.
%   Part 2: WarpFactory verifyTensor on the two clean reconstructed shells (C=10.44, C=21.18).
%   Part 3: grid convergence of rho_peak at N=24/32/48 for the retained rung.
% Requires WarpFactory (external; set WARPFACTORY_PATH; pinned to commit 03b10cb0). CPU only.
    here = fileparts(mfilename('fullpath'));
    wf = getenv('WARPFACTORY_PATH');
    assert(~isempty(wf), 'set WARPFACTORY_PATH to your WarpFactory checkout (commit 03b10cb0)');
    addpath(genpath(wf));
    d = load(fullfile(here, 'certificate.mat'));
    Aeq=d.Aeq; Aineq=d.Aineq; ell=double(d.ell(:)); b_ub=double(d.b_ub(:));
    B=double(d.B); NORM=double(d.NORM); C_FULL=double(d.C_FULL); N=size(Aeq,2);
    fid=fopen(fullfile(here,'MATLAB_REPRODUCTION.txt'),'w');
    log(fid, sprintf('WarpShell MATLAB/WarpFactory reproduction (repo data)  MATLAB %s', version));

    % ---- Part 1: C_LP cross-solve on the repo's matrices ----
    opts=optimoptions('linprog','Display','none','Algorithm','interior-point');
    [~,fval,flag]=linprog(-ell,Aineq,b_ub,Aeq,zeros(size(Aeq,1),1),-B*ones(N,1),B*ones(N,1),opts);
    C_LP=NORM*(-fval);
    log(fid, sprintf('[Part 1: C_LP cross-solve] MATLAB=%.6f  HiGHS=%.9f  rel.diff=%.2e  exitflag=%d  (frozen v18: 5.825968, 1.4e-5)', ...
        C_LP, C_FULL, abs(C_LP-C_FULL)/C_FULL, flag));

    % ---- Part 2: WarpFactory verifyTensor, two clean cases ----
    c_light=c(); G_grav=G(); Msun=1.989e30;
    cases={'clean_joint_C10.44',10.44,309.5,1.7466e15; 'clean_retained_C21.18',21.18,217.3,3.5433e15};
    for k=1:size(cases,1)
        nm=cases{k,1}; R_km=cases{k,3}; rho_frozen=cases{k,4};
        [vstr,htt,rho,M]=recon(R_km,c_light,G_grav,24);
        log(fid, sprintf('[Part 2: %s] verifyTensor=%s  h_tt_max=%.4e  rho_peak=%.4e  (frozen %.4e, rel.diff %.1e)  M=%.2f Msun', ...
            nm, vstr, htt, rho, rho_frozen, abs(rho-rho_frozen)/rho_frozen, M/Msun));
    end

    % ---- Part 3: grid convergence for the retained rung C=21.18 ----
    frozen=[3.543299e15,3.171132e15,3.642097e15]; jj=1;
    for Nn=[24 32 48]
        [vstr,htt,rho,~]=recon(217.3,c_light,G_grav,Nn);
        log(fid, sprintf('[Part 3: grid N=%2d] verifyTensor=%s  h_tt_max=%.4e  rho_peak=%.6e  (frozen %.6e, rel.diff %.1e)', ...
            Nn, vstr, htt, rho, frozen(jj), abs(rho-frozen(jj))/frozen(jj))); jj=jj+1;
    end
    log(fid,'DONE'); fclose(fid);
end

function [vstr,htt,rho,M]=recon(R_km,c_light,G_grav,Nn)
    R2=R_km*1000; R1=0.5*R2; u=0.20; M=u*c_light^2*R2/G_grav; ext=3.0*R2;
    metric=metricGet_WarpShellComoving([1,Nn,Nn,Nn],[0,ext/2,ext/2,ext/2],M,R1,R2,0,0,1,0,0,[1.0,ext/Nn,ext/Nn,ext/Nn]);
    vp=verifyTensor(metric,1); vstr='FAIL'; if vp, vstr='PASS'; end
    et=getEnergyTensor(metric,0,'fourth'); ci=round(Nn/2);
    rT00=squeeze(et.tensor{1,1}(1,ci,ci,:)); rgtt=squeeze(metric.tensor{1,1}(1,ci,ci,:));
    htt=max(abs(-1-rgtt(~isnan(rgtt)))); rho=max(abs(rT00(~isnan(rT00))))/c_light^2;
end

function log(fid,s); fprintf('%s\n',s); fprintf(fid,'%s\n',s); end
