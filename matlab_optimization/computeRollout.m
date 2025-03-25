function [states_rough, t_rough] = computeRollout(x0, t0, dt_dyn, N_dyn, omega_l, omega_r,  params)

    %init
    states_rough = zeros(3, N_dyn);
    t_rough = zeros(1, N_dyn);
          

    if  (params.int_steps ==0)
        [~,~,states_rough, t_rough] = integrate_dynamics(x0, 0,dt_dyn, N_dyn, omega_l, omega_r, params);
    else
        dt_step = dt_dyn/cast(params.int_steps-1, "double");
        for i=1:N_dyn              
            if (i>=2)     
              [states_rough(:,i), t_rough(i)] = integrate_dynamics(states_rough(:,i-1), t_rough(i-1), dt_step, params.int_steps, ...
                                                omega_l(i-1)*ones(1,params.int_steps), omega_r(i-1)*ones(1,params.int_steps), params); % keep Fr constant           
            else
              states_rough(:,i) = x0;
              t_rough(i) = t0;      
            end    
        end
    end




end