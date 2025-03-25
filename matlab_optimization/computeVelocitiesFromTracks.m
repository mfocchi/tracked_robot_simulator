function [v_input,omega_input] =  computeVelocitiesFromTracks(omega_l, omega_r, params)

    omega_wheel_l = omega_l; % [rad/s]
    omega_wheel_r = omega_r; % [rad/s]
    r = params.sprocket_radius;
    B = params.width;
    v_input = r * (omega_wheel_r + omega_wheel_l) / 2.0;
    omega_input = r * (omega_wheel_r - omega_wheel_l) / B;

end