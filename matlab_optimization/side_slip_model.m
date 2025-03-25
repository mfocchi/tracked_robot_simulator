function dxdt = side_slip_model(x, omega_l, omega_r, params)
 % Define persistent neural network model
    persistent alpha_model_forcodegen_01;
persistent alpha_model_forcodegen_04;
    %omega_l/r are the unicycle wheel speed
    theta = x(3);    
    % ideal wheel speed in rad/s
    r = params.sprocket_radius;
    B = params.width; 
    %     % ideal linear and angular velocity
    v_input = r * (omega_r + omega_l) / 2.0;
    omega_input = r * (omega_r - omega_l) / B;

    if strcmp(params.side_slippage_estimation,'EXP')    
        if omega_input == 0 &&  v_input~=0
            turning_radius_input =  1e08*sign(v_input);
        elseif omega_input == 0 &&  v_input==0
    
            turning_radius_input = 1e8;
        else
             turning_radius_input = v_input / omega_input;
        end
    
        % side slip        
        if(turning_radius_input > 0.0) % turning left 
            alpha = params.side_slip_angle_coefficients_left(1)*exp(params.side_slip_angle_coefficients_left(2)*turning_radius_input);
        else % turning right
            % inverting the slips with respect to test results   
            alpha = params.side_slip_angle_coefficients_right(1)*exp(params.side_slip_angle_coefficients_right(2)*turning_radius_input);
        end
    elseif strcmp(params.side_slippage_estimation,'NET')  
        %%TOO SLOW
        % Create the input array for the models
        %input_data = py.list([omega_l, omega_r]);
        % Predict using model_beta_l
        %prediction_beta_l = double(params.model_beta_l.predict(input_data));
        % Predict using model_beta_r
        %prediction_beta_r = double(params.model_beta_r.predict(input_data));
        % Predict using model_alpha
        %alpha = double(params.model_alpha.predict(input_data));
        
        %STILL SLOW (0.23s) CALL DIRECTLY BINARY
        % cd('./eval_slippage/cpp/');
        % alpha = call_binary(omega_l, omega_r);
        % cd('../../');

        %%FAST (0.0020s) use martlab model trained with regressorLearner
         if isempty(alpha_model_forcodegen_01)
            % Load the model only once      
            alpha_model_forcodegen_01 = loadLearnerForCoder('matlabNN/friction_0.1/alpha_model_forcodegen');                      
            alpha_model_forcodegen_04 = loadLearnerForCoder('matlabNN/friction_0.4/alpha_model_forcodegen');
         end
         if params.friction_coeff == 0.1
             alpha = predict(alpha_model_forcodegen_01, [omega_l, omega_r]);
         else %0.4
             alpha = predict(alpha_model_forcodegen_04, [omega_l, omega_r]);
         end
       
    else
       alpha = 0;
       disp('wrong side slippage estimation setting')
    end
    
    dxdt = [v_input*(cos(theta)-sin(theta)*tan(alpha)); v_input*(sin(theta)+cos(theta)*tan(alpha)); omega_input];
end

