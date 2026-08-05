%% Goswami compass-gait period-doubling trajectories
% Run this script directly. It creates CSV files, diagnostic PNGs, and
% optional MP4 animations for period 1, 2, 4, 8, and chaos.
%
% State convention:
%   x = [theta_ns; theta_s; dtheta_ns; dtheta_s]
% Angles are in radians and are measured counterclockwise from vertical.

clear; close all; clc;

OUT = 'compass_goswami_output_matlab';
MAKE_VIDEO = true;             % set false for CSV + plots only
DT = 0.01;                     % sampling interval for saved trajectories
SELECT = {'period1','period2','period4','period8','chaos'};

if ~exist(OUT, 'dir')
    mkdir(OUT);
end

% Nominal robot: total mass 20 kg, mu=2, beta=1, l=1 m.
p.m = 5.0;
p.mH = 10.0;
p.l = 1.0;
p.a = 0.5;
p.b = 0.5;
p.g = 9.81;

cases = goswami_cases();
allResults = struct();

for i = 1:numel(SELECT)
    label = SELECT{i};
    cfg = cases.(label);
    fprintf('Simulating %s at %.2f deg ...\n', label, cfg.phiDeg);

    [segments, R] = simulate_compass(cfg.phiDeg, cfg.x0, ...
        cfg.nSteps, DT, p);
    allResults.(label).segments = segments;
    allResults.(label).R = R;
    allResults.(label).phiDeg = cfg.phiDeg;
    allResults.(label).period = cfg.period;

    save_csv_files(OUT, label, segments, R, p);
    plot_diagnostics(OUT, label, cfg.phiDeg, segments, R);
    if MAKE_VIDEO
        save_walking_video(OUT, label, cfg.phiDeg, segments, p, DT);
    end

    disp('  last step periods:');
    i0 = max(1, numel(R.stepPeriod)-7);
    disp(R.stepPeriod(i0:end).');
end

required = {'period1','period2','period4','period8','chaos'};
if all(isfield(allResults, required))
    plot_figure10_atlas(OUT, allResults);
end


function cases = goswami_cases()
% Every initial state is immediately post-impact.

cases.period1.phiDeg = 4.00;
cases.period1.period = 1;
cases.period1.x0 = [-0.368711604132;  0.229085263972; ...
                    -0.216411423691; -1.139899112182];
cases.period1.nSteps = 40;

cases.period2.phiDeg = 4.75;
cases.period2.period = 2;
cases.period2.x0 = [-0.381789947562;  0.215983668622; ...
                    -0.153332476821; -1.153674078371];
cases.period2.nSteps = 40;

cases.period4.phiDeg = 5.00;
cases.period4.period = 4;
cases.period4.x0 = [-0.392348120036;  0.217815194837; ...
                    -0.108006761939; -1.161444888863];
cases.period4.nSteps = 48;

cases.period8.phiDeg = 5.02;
cases.period8.period = 8;
cases.period8.x0 = [-0.394646896303;  0.219415839402; ...
                    -0.097321131256; -1.162198996812];
cases.period8.nSteps = 64;

cases.chaos.phiDeg = 5.20;
cases.chaos.period = NaN;
cases.chaos.x0 = [-0.401092708453;  0.219578466246; ...
                  -0.087963122288; -1.170237773352];
cases.chaos.nSteps = 200;
end


function [segments, R] = simulate_compass(phiDeg, x0, nSteps, dt, p)
phi = deg2rad(phiDeg);
x = x0(:);
tGlobal = 0.0;
support = [0.0; 0.0];
segments = cell(nSteps, 1);

R.step = (1:nSteps).';
R.impactTime = zeros(nSteps, 1);
R.stepPeriod = zeros(nSteps, 1);
R.x = zeros(nSteps, 4);

opts = odeset('RelTol',1e-10, 'AbsTol',1e-12, 'MaxStep',0.005, ...
    'Events',@(t,z) guard_event(t,z,phi));

for k = 1:nSteps
    sol = ode45(@(t,z) vector_field(t,z,p), [0.0 5.0], x, opts);
    if isempty(sol.xe)
        error('No foot strike found at step %d.', k);
    end

    T = sol.xe(end);
    tLocal = (0.0:dt:T).';
    if isempty(tLocal) || tLocal(end) < T
        tLocal = [tLocal; T]; %#ok<AGROW>
    else
        tLocal(end) = T;
    end
    X = deval(sol, tLocal).';
    xMinus = sol.ye(:,end);
    X(end,:) = xMinus.';

    seg.step = k - 1;         % zero-based, matching the Python CSV
    seg.t = tGlobal + tLocal;
    seg.x = X;
    seg.support = support;
    segments{k} = seg;

    xPlus = reset_map(xMinus, p);
    tGlobal = tGlobal + T;
    R.impactTime(k) = tGlobal;
    R.stepPeriod(k) = T;
    R.x(k,:) = xPlus.';

    % The old swing-foot impact point is the next support-foot origin.
    [~, newSupport, ~] = geometry(xMinus, support, p);
    support = newSupport;
    x = xPlus;
end
end


function dx = vector_field(~, x, p)
thNS = x(1); thS = x(2);
dthNS = x(3); dthS = x(4);
delta = thS - thNS;

M = [p.m*p.b^2, -p.m*p.l*p.b*cos(delta); ...
    -p.m*p.l*p.b*cos(delta), (p.mH+p.m)*p.l^2 + p.m*p.a^2];
N = [0.0, p.m*p.l*p.b*sin(delta)*dthS; ...
    -p.m*p.l*p.b*sin(delta)*dthNS, 0.0];
G = [p.m*p.b*p.g*sin(thNS); ...
    -(p.mH*p.l + p.m*p.a + p.m*p.l)*p.g*sin(thS)];

dq = [dthNS; dthS];
ddq = M \ (-(N*dq + G));
dx = [dq; ddq];
end


function [value, isterminal, direction] = guard_event(~, x, phi)
% Foot strike: theta_ns + theta_s = -2*phi, swing leg ahead of stance.
if x(1) - x(2) > 0.01
    value = x(1) + x(2) + 2.0*phi;
else
    % Suppress a spurious event immediately after the role swap.
    value = 1.0;
end
isterminal = 1;
direction = -1;
end


function xPlus = reset_map(xMinus, p)
thNS = xMinus(1); thS = xMinus(2);
dthNS = xMinus(3); dthS = xMinus(4);
c = cos(thS - thNS);          % cos(2*alpha)

Qminus = [-p.m*p.a*p.b, ...
    (p.mH*p.l^2 + 2*p.m*p.a*p.l)*c - p.m*p.a*p.b; ...
    0.0, -p.m*p.a*p.b];

Qplus = [p.m*p.b*(p.b-p.l*c), ...
    p.m*p.l*(p.l-p.b*c) + p.mH*p.l^2 + p.m*p.a^2; ...
    p.m*p.b^2, -p.m*p.b*p.l*c];

dqPlus = Qplus \ (Qminus*[dthNS; dthS]);
xPlus = [thS; thNS; dqPlus];  % leg-role swap
end


function [hip, swing, support] = geometry(x, supportXY, p)
support = supportXY(:);
thNS = x(1); thS = x(2);
hip = support + [-p.l*sin(thS); p.l*cos(thS)];
swing = hip + [p.l*sin(thNS); -p.l*cos(thNS)];
end


function save_csv_files(outDir, label, segments, R, p)
rows = [];
for k = 1:numel(segments)
    seg = segments{k};
    n = numel(seg.t);
    event = zeros(n,1);       % 0 flow, +1 post-impact, -1 pre-impact
    event(1) = 1;
    event(end) = -1;
    geom = zeros(n,6);
    for j = 1:n
        [hip,swing,support] = geometry(seg.x(j,:).', seg.support, p);
        geom(j,:) = [hip.', swing.', support.'];
    end
    rows = [rows; seg.t, repmat(seg.step,n,1), event, seg.x, geom]; %#ok<AGROW>
end

names = {'t','step','event','theta_ns','theta_s','dtheta_ns','dtheta_s', ...
    'hip_x','hip_y','swing_x','swing_y','support_x','support_y'};
Tseries = array2table(rows, 'VariableNames', names);
writetable(Tseries, fullfile(outDir, [label '_timeseries.csv']));

retRows = [R.step, R.impactTime, R.stepPeriod, R.x];
retNames = {'step','impact_time','step_period','theta_ns','theta_s', ...
    'dtheta_ns','dtheta_s'};
Treturns = array2table(retRows, 'VariableNames', retNames);
writetable(Treturns, fullfile(outDir, [label '_returns.csv']));
end


function plot_diagnostics(outDir, label, phiDeg, segments, R)
t = []; X = [];
for k = 1:numel(segments)
    t = [t; segments{k}.t]; %#ok<AGROW>
    X = [X; segments{k}.x]; %#ok<AGROW>
end

fig = figure('Visible','off','Color','w','Position',[100 100 1000 700]);
subplot(2,2,1);
plot(t,X(:,1),'LineWidth',1.0); hold on;
plot(t,X(:,2),'LineWidth',1.0);
xlabel('time (s)'); ylabel('angle (rad)');
legend('\theta_{ns}','\theta_s','Location','best'); grid on;

subplot(2,2,2);
plot(t,X(:,3),'LineWidth',1.0); hold on;
plot(t,X(:,4),'LineWidth',1.0);
xlabel('time (s)'); ylabel('angular velocity (rad/s)');
legend('d\theta_{ns}/dt','d\theta_s/dt','Location','best'); grid on;

subplot(2,2,3);
plot(R.step,R.stepPeriod,'.-','MarkerSize',8);
xlabel('step number'); ylabel('step period (s)'); grid on;

subplot(2,2,4);
plot(R.x(1:end-1,1),R.x(2:end,1),'.','MarkerSize',8); hold on;
lo = min(R.x(:,1)); hi = max(R.x(:,1));
pad = max(0.002, 0.05*(hi-lo));
plot([lo-pad hi+pad],[lo-pad hi+pad],'k--','LineWidth',0.8);
xlim([lo-pad hi+pad]); ylim([lo-pad hi+pad]); axis square; grid on;
xlabel('\theta_{ns,k}'); ylabel('\theta_{ns,k+1}');

sgtitle(sprintf('%s: compass gait at \\phi = %.2f deg',label,phiDeg));
print(fig, fullfile(outDir,[label '_diagnostics.png']), '-dpng','-r180');
close(fig);
end


function plot_figure10_atlas(outDir, results)
% Modern counterpart of Goswami et al. Figure 10, including period 1.
order = {'period1','period2','period4','period8','chaos'};
names = {'Period 1','Period 2','Period 4','Period 8','Chaotic gait'};
letters = {'(a)','(b)','(c)','(d)','(e)'};

fig = figure('Visible','off','Color','w','Position',[50 50 1400 850]);
tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');

for i = 1:numel(order)
    label = order{i};
    item = results.(label);
    ax = nexttile;
    if strcmp(label,'chaos')
        nShow = 100;
    else
        nShow = max(2,item.period);
    end
    plot_physical_leg_phase(ax,item.segments,item.R,nShow);
    title(ax,sprintf('%s %s  |  \\phi = %.2f deg', ...
        letters{i},names{i},item.phiDeg),'FontSize',12);
    if i == 1
        legend(ax,{'physical leg A','physical leg B'}, ...
            'Location','northeast','FontSize',9);
    end
end

axText = nexttile;
axis(axText,'off');
text(axText,0.03,0.88,{ ...
    'Compass-gait period-doubling cascade','', ...
    'Nominal model: \mu = 2, \beta = 1, l = 1 m','', ...
    'Solid curves: continuous swing dynamics', ...
    'Dashed curves: instantaneous impact reset','', ...
    'Colors follow physical legs across the', ...
    'stance/nonstance label swap.'}, ...
    'VerticalAlignment','top','FontSize',12,'Interpreter','tex');

sgtitle(fig,'Passive Compass Gait: Phase-Plane Period Doubling', ...
    'FontSize',17,'FontWeight','bold');
print(fig,fullfile(outDir,'figure10_modern_phase_portraits.png'), ...
    '-dpng','-r220');
print(fig,fullfile(outDir,'figure10_modern_phase_portraits.pdf'), ...
    '-dpdf','-painters');
close(fig);
end


function plot_physical_leg_phase(ax,segments,R,nShow)
% Track physical leg identity across each stance/nonstance role swap.
blue = [0.000 0.447 0.741];
orange = [0.850 0.325 0.098];
blueDash = 0.65*blue + 0.35*[1 1 1];
orangeDash = 0.65*orange + 0.35*[1 1 1];
nShow = min([nShow,numel(segments),size(R.x,1)]);
hold(ax,'on');

for k = 1:nShow
    X = segments{k}.x;
    xMinus = X(end,:).';
    xPlus = R.x(k,:).';
    nsIsA = mod(k-1,2) == 0;

    if nsIsA
        aCurve = X(:,[1 3]); bCurve = X(:,[2 4]);
        aPre = xMinus([1 3]); aPost = xPlus([2 4]);
        bPre = xMinus([2 4]); bPost = xPlus([1 3]);
    else
        aCurve = X(:,[2 4]); bCurve = X(:,[1 3]);
        aPre = xMinus([2 4]); aPost = xPlus([1 3]);
        bPre = xMinus([1 3]); bPost = xPlus([2 4]);
    end

    plot(ax,aCurve(:,1),aCurve(:,2),'Color',blue,'LineWidth',1.25);
    plot(ax,bCurve(:,1),bCurve(:,2),'Color',orange,'LineWidth',1.25);
    plot(ax,[aPre(1) aPost(1)],[aPre(2) aPost(2)],'--', ...
        'Color',blueDash,'LineWidth',0.8);
    plot(ax,[bPre(1) bPost(1)],[bPre(2) bPost(2)],'--', ...
        'Color',orangeDash,'LineWidth',0.8);
end

xlim(ax,[-0.48 0.58]); ylim(ax,[-3.10 2.90]);
grid(ax,'on'); ax.GridColor = [0.82 0.82 0.82]; ax.GridAlpha = 0.65;
xlabel(ax,'angular position \theta (rad)');
ylabel(ax,'angular velocity d\theta/dt (rad/s)');
end


function save_walking_video(outDir, label, phiDeg, segments, p, dt)
% Give every gait the same 12.00-second, uniformly sampled timeline.
videoDuration = 12.0;
videoFPS = 50;
writer = VideoWriter(fullfile(outDir,[label '.mp4']), 'MPEG-4');
writer.FrameRate = videoFPS;
open(writer);

fig = figure('Color','w','Position',[100 100 800 450]);
ax = axes(fig); hold(ax,'on'); axis(ax,'equal'); grid(ax,'on');
stanceLine = plot(ax,nan,nan,'o-','LineWidth',3,'Color',[0 0.447 0.741]);
swingLine = plot(ax,nan,nan,'o-','LineWidth',3,'Color',[0.850 0.325 0.098]);
groundLine = plot(ax,nan,nan,'k-','LineWidth',1);
xlabel(ax,'horizontal position (m)'); ylabel(ax,'height (m)');
legend(ax,{'stance leg','swing leg'},'Location','northeast');
phi = deg2rad(phiDeg);
targetTimes = linspace(0.0,videoDuration,round(videoDuration*videoFPS)+1);
segIndex = 1;

for t = targetTimes
    while segIndex < numel(segments) && ...
            t > segments{segIndex}.t(end) + 1e-12
        segIndex = segIndex + 1;
    end
    seg = segments{segIndex};
    if t > seg.t(end) + 1e-9
        error('Trajectory for %s is shorter than %.2f s.',label,videoDuration);
    end
    x = interp1(seg.t,seg.x,t,'linear').';
    [hip,swing,support] = geometry(x,seg.support,p);
    center = hip(1);
    gx = [center-2.0, center+2.0];
    gy = -tan(phi)*gx;
    set(groundLine,'XData',gx,'YData',gy);
    set(stanceLine,'XData',[support(1),hip(1)], ...
        'YData',[support(2),hip(2)]);
    set(swingLine,'XData',[hip(1),swing(1)], ...
        'YData',[hip(2),swing(2)]);
    xlim(ax,[center-1.4 center+1.4]);
    groundYAtCenter = -tan(phi)*center;
    ylim(ax,[groundYAtCenter-0.25 groundYAtCenter+1.25]);
    title(ax,sprintf('%s, slope %.2f deg, t = %.2f s',label,phiDeg,t));
    drawnow;
    writeVideo(writer,getframe(fig));
end

close(writer);
close(fig);
end
