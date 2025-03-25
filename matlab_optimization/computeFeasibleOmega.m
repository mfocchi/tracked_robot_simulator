function [omega_min, omega_max] = computeFeasibleOmega(long_v, params)
    if nargin<2
        
        params.width = 0.606;
        params.sprocket_radius = 0.0856;
        params.omega_w_max =  6.0795; % it cooresponds to 209 rad/s before gearbox
    end   

    % these are computations after the gearbox
    C = [-params.sprocket_radius/( params.width); params.sprocket_radius/(params.width)];
    Aeq = [params.sprocket_radius/(2 ), params.sprocket_radius/(2 )];
    beq = long_v;

    lb = [-params.omega_w_max -params.omega_w_max];
    ub = [params.omega_w_max, params.omega_w_max];

    options = optimoptions('linprog','Display','none'); %silencing ouptput
    [x, feval] = linprog(C,[],[],Aeq,beq,lb,ub, options);
    omega_max = -feval;
    [x, feval] = linprog(-C,[],[],Aeq,beq,lb,ub, options);
    omega_min = feval;
end

