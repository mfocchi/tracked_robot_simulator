function [turning_radius_input, beta_l, beta_r, alpha] = evalSlippage(wheel_l, wheel_r, params)


r = params.sprocket_radius;
B = params.width;
v_input = r * (wheel_l + wheel_r) / 2.0;
omega_input = r * (wheel_r - wheel_l) / B;


turning_radius_input = [];
beta_l = [];
beta_r = [];


for i=1:length(omega_input)
    if omega_input(i) == 0 &&  v_input(i)~=0
        turning_radius_input(i) =  1e08*sign(v_input(i));
    elseif omega_input(i) == 0 &&  v_input(i)==0
        turning_radius_input(i) = 1e8;
    else
        turning_radius_input(i) = v_input(i) / omega_input(i);
    end
    %compute track velocity from encoder
    v_enc_l = r*wheel_l(i);
    v_enc_r = r*wheel_r(i);

    if strcmp(params.side_slippage_estimation,'EXP')
       

        % side slip
        if(turning_radius_input(i) >= 0.0) % turning left
            alpha(i) = params.side_slip_angle_coefficients_left(1)*exp(params.side_slip_angle_coefficients_left(2)*turning_radius_input(i));
        else % turning right
            % inverting the slips with respect to test results
            alpha(i) = params.side_slip_angle_coefficients_right(1)*exp(params.side_slip_angle_coefficients_right(2)*turning_radius_input(i));
        end
        if strcmp(params.model, 'LONGSIDE')
            %long slip
            %estimate beta_inner, beta_outer from turning radius
            if(turning_radius_input(i) >= 0.0)% turning left, positive radius, left wheel is inner right wheel is outer
                beta_l(i) = params.beta_slip_inner_coefficients_left(1)*exp(params.beta_slip_inner_coefficients_left(2)*turning_radius_input(i));
                % the inner is accelerated the outer slowed down
                beta_r(i) = params.beta_slip_outer_coefficients_left(1)*exp(params.beta_slip_outer_coefficients_left(2)*turning_radius_input(i));
            else % turning right, negative radius, left wheel is outer right is inner
                beta_r(i) = params.beta_slip_inner_coefficients_right(1)*exp(params.beta_slip_inner_coefficients_right(2)*turning_radius_input(i));
                beta_l(i) =  params.beta_slip_outer_coefficients_right(1)*exp(params.beta_slip_outer_coefficients_right(2)*turning_radius_input(i));
            end
        else
            beta_l(i) = 0;
            beta_r(i) = 0;
        end

    elseif strcmp(params.side_slippage_estimation,'NET')


        alpha_model_forcodegen = loadLearnerForCoder('alpha_model_forcodegen');
        alpha(i) = predict(alpha_model_forcodegen, [wheel_l(i), wheel_r(i)]);
        
        if strcmp(params.model, 'LONGSIDE')
            beta_l_model_forcodegen = loadLearnerForCoder('beta_l_model_forcodegen');
            beta_l(i) = predict(beta_l_model_forcodegen, [wheel_l(i), wheel_r(i)]);
    
            beta_r_model_forcodegen = loadLearnerForCoder('beta_r_model_forcodegen');
            beta_r(i) = predict(beta_r_model_forcodegen, [wheel_l(i), wheel_r(i)]);
        else
            beta_l(i) = 0;
            beta_r(i) = 0;
        end
    end

end