function input_table = extractUnicycleSetpoint(sol, dt)
    tf = sol.time(end);
    new_times = 0:dt:tf;
    new_v = interp1(sol.time, sol.v_input, new_times);
    new_omega = interp1(sol.time, sol.omega_input, new_times);
    new_x     = interp1(sol.time, sol.p(1,:), new_times);
    new_y     = interp1(sol.time, sol.p(2,:), new_times);
    new_theta = interp1(sol.time, sol.p(3,:), new_times);

    input_table = table(new_times',new_v', new_omega',new_x',new_y',new_theta', 'VariableNames',["t","v","omega","x","y","theta"]);

end

