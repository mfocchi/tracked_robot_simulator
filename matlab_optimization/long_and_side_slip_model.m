function dxdt = long_and_side_slip_model(x, omega_l, omega_r, params)
    persistent alpha_model_forcodegen_01;
    persistent beta_l_model_forcodegen_01;
    persistent beta_r_model_forcodegen_01;
    persistent alpha_model_forcodegen_04;
    persistent beta_l_model_forcodegen_04;
    persistent beta_r_model_forcodegen_04;
    %omega_l/r are the unicycle wheel speed
    theta = x(3);    
    % ideal wheel speed in rad/s
    r = params.sprocket_radius;
    B = params.width; 
    %     % ideal linear and angular velocity
    v_input = r * (omega_r + omega_l) / 2.0;
    omega_input = r * (omega_r - omega_l) / B;
    %compute track velocity from encoder
    v_enc_l = r*omega_l;
    v_enc_r = r*omega_r;
    
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
        
            %long slip    
        
            %estimate beta_inner, beta_outer from turning radius
            if(turning_radius_input >= 0.0)% turning left, positive radius, left wheel is inner right wheel is outer
                beta_l = params.beta_slip_inner_coefficients_left(1)*exp(params.beta_slip_inner_coefficients_left(2)*turning_radius_input);
                % the inner is accelerated the outer slowed down
                beta_r = params.beta_slip_outer_coefficients_left(1)*exp(params.beta_slip_outer_coefficients_left(2)*turning_radius_input);
            else % turning right, negative radius, left wheel is outer right is inner
                beta_r = params.beta_slip_inner_coefficients_right(1)*exp(params.beta_slip_inner_coefficients_right(2)*turning_radius_input);     
                beta_l =  params.beta_slip_outer_coefficients_right(1)*exp(params.beta_slip_outer_coefficients_right(2)*turning_radius_input);
            end

        
    elseif strcmp(params.side_slippage_estimation,'NET')
         
         % Load the model only once
         if isempty(alpha_model_forcodegen_01)
            % Load the model only once      
            alpha_model_forcodegen_01 = loadLearnerForCoder('matlabNN/friction_0.1/alpha_model_forcodegen');                      
            alpha_model_forcodegen_04 = loadLearnerForCoder('matlabNN/friction_0.4/alpha_model_forcodegen');
         end

         if isempty(beta_l_model_forcodegen_01)
            % Load the model only once      
            beta_l_model_forcodegen_01 = loadLearnerForCoder('matlabNN/friction_0.1/beta_l_model_forcodegen');                      
            beta_l_model_forcodegen_04 = loadLearnerForCoder('matlabNN/friction_0.4/beta_l_model_forcodegen');
         end

         if isempty(beta_r_model_forcodegen_01)
            % Load the model only once      
            beta_r_model_forcodegen_01 = loadLearnerForCoder('matlabNN/friction_0.1/beta_r_model_forcodegen');                      
            beta_r_model_forcodegen_04 = loadLearnerForCoder('matlabNN/friction_0.4/beta_r_model_forcodegen');
         end


        %%FAST (0.0020s) use martlab model trained with regressorLearner
        if params.friction_coeff == 0.1
            alpha = predict(alpha_model_forcodegen_01, [omega_l, omega_r]);
        else %0.4
            alpha = predict(alpha_model_forcodegen_04, [omega_l, omega_r]);
        end
        %%FAST (0.0020s) use martlab model trained with regressorLearner
        if params.friction_coeff == 0.1
            beta_l = predict(beta_l_model_forcodegen_01, [omega_l, omega_r]);
        else %0.4
            beta_l = predict(beta_l_model_forcodegen_04, [omega_l, omega_r]);
        end

        %%FAST (0.0020s) use martlab model trained with regressorLearner
        if params.friction_coeff == 0.1
            beta_r = predict(beta_r_model_forcodegen_01, [omega_l, omega_r]);
        else %0.4
            beta_r = predict(beta_r_model_forcodegen_04, [omega_l, omega_r]);
        end
    else
       alpha = 0;
       beta_l = 0;
       beta_r = 0;
       disp('wrong side slippage estimation setting')
    end


    v_enc_l=v_enc_l+beta_l;
    v_enc_r=v_enc_r+beta_r;

    omega_l_s = 1/r * v_enc_l;
    omega_r_s = 1/r * v_enc_r;

    % actual linear and angular velocity
    v_input_s     = r * (omega_r_s + omega_l_s) / 2;
    omega_input_s = r * (omega_r_s - omega_l_s) / B;
         
    dxdt = [v_input_s*(cos(theta)-sin(theta)*tan(alpha)); v_input_s*(sin(theta)+cos(theta)*tan(alpha)); omega_input_s];
end

