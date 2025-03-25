function plot_dubins(p0, pf, params)
    
    %luigi
    % curvature_max = params.omega_max/params.v_max;
    % dubinsObj= dubinsClass;
    % %Find optimal Dubins solution
    % [pidx, curve] = dubinsObj.dubins_shortest_path(p0(1), p0(2), p0(3), pf(1), pf(2), pf(3), curvature_max);
    % %plot dubin
    % dubinsObj.plotdubins(curve, true, [1 0 0], [0 0 0], [1 0 0]); hold on;
    % 
    % 
    %matlab plot after integration
    dubConnObj = dubinsConnection;
    curvature_max = params.omega_max/params.v_max; %using max forward speed
    dubConnObj.MinTurningRadius = 1/curvature_max;
    [pathSegObj, pathCosts] = connect(dubConnObj,p0',pf');
    omegas = get_omega_from_dubins(pathSegObj{1}, params.v_max,dubConnObj.MinTurningRadius);
    [omega_l0, omega_r0, time] = getWheelVelocityParamsFromDubin(params, pathSegObj{1}.MotionLengths, omegas, params.dt);
    numbe_of_samples = length(omega_l0);
    params.int_steps = 0;
    params.model = 'UNICYCLE'; %need to set unicycle!!!
    %integrate the fine grid omegas
    [states, t] = computeRollout(p0, 0,params.dt, numbe_of_samples, omega_l0, omega_r0, params);
    x = states(1,:);
    y = states(2,:);
    theta = states(3,:);
    p =  [x; y; theta];
    color_input = 'k'; 
    % actual dubins traj obtained by fine integration (black line)
    plot(p(1,:), p(2,:), '-.', 'Color', color_input, 'linewidth', 2) ;
end