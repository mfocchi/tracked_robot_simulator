function [cost, cost_components]= cost(x, p0,  pf, params)

    Tf =  x(1);
    omega_l = x(params.num_params+1:params.num_params+params.N_dyn); 
    omega_r = x(params.num_params+params.N_dyn+1:params.num_params+2*params.N_dyn); 

    
    
    % check they are column vectors
    p0 = p0(:);
    pf = pf(:);

    dt_dyn = Tf / (params.N_dyn-1); 
    
    %single shooting
    [states, t] = computeRollout(p0, 0,dt_dyn, params.N_dyn, omega_l, omega_r, params);
    x = states(1,:);
    y = states(2,:);
    theta = states(3,:);
    
    p_0 = [x(:,1); y(:,1); theta(:,1)];
    p_f = [x(:,end); y(:,end); theta(:,end)];


    xy_der= diff(x)./ diff(y);
    theta_der = diff(theta);
    smoothing_xy_der =  sum(xy_der.^2);
    smoothing_theta_der= sum(theta_der.^2);
    
    [v_input,omega_input] =  computeVelocitiesFromTracks(omega_l, omega_r, params);
    tracking = norm(p_f - pf);
    
    
    %smooting on wheels (temporal derivative) (not used)
    %smoothing_wheels = sum(diff(omega_l).^2)+ sum(diff(omega_l).^2);  
    %smoothing on speed (temporal derivative)
    smoothing_speed = sum(diff(v_input).^2)+ sum(diff(omega_input).^2);
    
    cost_components = struct;
    %to avoid nan issue incorporate them in the cost only if the weight is
    %not zero!
    if  params.w1 ~=0 
        cost_components.time   =  params.w1 * Tf;
    else 
        cost_components.time   = 0;
    end

    if  params.w2 ~=0     
        cost_components.smoothing_speed   =  params.w2 *smoothing_speed;
    else 
        cost_components.smoothing_speed   = 0;
    end

    if params.w3~=0
        cost_components.smoothing_xy_der   =  params.w3 *smoothing_xy_der;
    else 
        cost_components.smoothing_xy_der   = 0;
    end

    if params.w4~=0
        cost_components.smoothing_theta_der   =  params.w4 *smoothing_theta_der;
    else
        cost_components.smoothing_theta_der   = 0;
    end
    
        
    if params.DEBUG_COST

        fprintf('cost components: time: %f,  smoothing_speed : %f smoothing_xy_der : %f smoothing_theta_der : %f  \n \n',...
                        cost_components.time,  cost_components.smoothing_speed,  cost_components.smoothing_xy_der,  cost_components.smoothing_theta_der);
    end
    cost =  cost_components.time  +   cost_components.smoothing_speed +  cost_components.smoothing_xy_der+   cost_components.smoothing_theta_der ;
end