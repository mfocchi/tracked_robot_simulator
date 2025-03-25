function dxdt = unicycle_model(x, omega_l, omega_r, params)
    theta = x(3);    

    % ideal wheel speed in rad/s
    r = params.sprocket_radius;
    B = params.width; 
    %     % ideal linear and angular velocity
    v_input = r * (omega_r + omega_l) / 2.0;
    omega_input = r * (omega_r - omega_l) / B;

    dxdt = [v_input*cos(theta); v_input*sin(theta); omega_input]; 
end

