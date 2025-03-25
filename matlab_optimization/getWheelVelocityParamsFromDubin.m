function [omega_l0, omega_r0, time] = getWheelVelocityParamsFromDubin(params, lengths, omegas, dt) 
     % get total time 
    t0 = sum(lengths)/params.v_max; 

    switch_times = [cumsum(lengths) / params.v_max];
    % if not specified compute dt
    if nargin<4
        dt = t0 / (params.N_dyn );
    end

    % build  vector
    omega0 = [];
    v0 = [];
    omega_r0 = [];
    omega_l0 = [];
    t_ = 0.;
    time = [];
    while (t_<=t0)
        if  t_ < switch_times(1)
            omega =  omegas(1);
        elseif t_ <switch_times(2)
            omega = omegas(2);
        else 
            omega = omegas(3);
        end
        omega0 = [omega0 omega];
        v0 = [ v0 params.v_max];
        
        r = params.sprocket_radius;
        B = params.width;
        A = [r/2, r/2; r/B, -r/B];
        y = inv(A)*[params.v_max; omega];
        omega_r0 = [omega_r0 y(1)]; 
        omega_l0 = [omega_l0 y(2)];
        time = [time t_ ];
        t_ =  t_+ dt;
    end
end
