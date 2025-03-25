%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Demo usage of Dubins shortest path function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Reset environment
% close all; clear all; clc;
% 
% % Define problem data
% x0 = 0; 
% y0 = 0; 
% th0 = -2/3*pi;
% xf = 4; 
% yf = 0; 
% thf = pi/3.0;
% Kmax = 3.0;
% 
% 
% % Find optimal Dubins solution
% [pidx, curve] = dubins_shortest_path(x0, y0, th0, xf, yf, thf, Kmax);
% 
% % Plot and display solution if valid
% if pidx > 0
%   figure; axis equal;
%   plotdubins(curve, true, [1 0 0], [0 0 0], [1 0 0]);
%   curve.a1
%   curve.a2
%   curve.a3
%   curve.L
% else
%   fprintf('Failed!\n');
% end

 classdef dubinsClass
% 
%     
    methods
   
  % Plot a Dubins curve
    function plotdubins(obj,curve, decorations, c1, c2, c3)
      currhold = ishold;
      hold on;

      % Plot arcs
      plotarc(obj,curve.a1, c1);
      plotarc(obj,curve.a2, c2);
      plotarc(obj,curve.a3, c3);

      % Plot initial and final position and orientation
      if decorations
        plot(curve.a1.x0, curve.a1.y0, 'ob');
        plot(curve.a3.xf, curve.a3.yf, 'ob');
        quiver(curve.a1.x0, curve.a1.y0, 0.1*curve.L*cos(curve.a1.th0), 0.1*curve.L*sin(curve.a1.th0), 'b', 'LineWidth', 2);
        quiver(curve.a3.xf, curve.a3.yf, 0.1*curve.L*cos(curve.a3.thf), 0.1*curve.L*sin(curve.a3.thf), 'b', 'LineWidth', 2);
      end

      if ~currhold
        hold off;
      end
    end
    
    % Plot an arc (circular or straight) composing a Dubins curve
    function plotarc(obj,arc, color)
      npts = 100;
      pts = zeros(npts+1, 2);
      for j = 0:npts
        s = arc.L/npts * j;
        [x, y] = circline(s, arc.x0, arc.y0, arc.th0, arc.k);
        pts(j+1, 1:2) = [x, y];
      end
      plot(pts(:, 1), pts(:, 2), 'Color', color, 'LineWidth', 4);
    end

    
     function [pidx, curve, k, lengths] = dubins_shortest_path(obj,x0, y0, th0, xf, yf, thf, Kmax)
      % Compute params of standard scaled problem
      [sc_th0, sc_thf, sc_Kmax, lambda] = scaleToStandard(x0, y0, th0, xf, yf, thf, Kmax);

      % Define the functions corresponding to the different primitives, and the
      % corresponding curvatue signs 
      primitives = {@LSL, @RSR, @LSR, @RSL, @RLR, @LRL};
      ksigns = [
                  1  0  1 % LSL
                 -1  0 -1 % RSR
                  1  0 -1 % LSR
                 -1  0  1 % RSL
                 -1  1 -1 % RLR
                  1 -1  1 % LRL
               ];

      % Try all the possible primitives, to find the optimal solution
      pidx = -1;
      L = inf;
      for i = 1:numel(primitives)
        [ok, sc_s1_c, sc_s2_c, sc_s3_c] = primitives{i}(sc_th0, sc_thf, sc_Kmax);
        Lcur = sc_s1_c + sc_s2_c + sc_s3_c;
        if (ok && Lcur<L)
          L = Lcur;
          sc_s1 = sc_s1_c;
          sc_s2 = sc_s2_c;
          sc_s3 = sc_s3_c;
          pidx = i;
        end
      end

      curve = [];
      if pidx > 0
        % Transform the solution to the problem in standard form to the 
        % solution of the original problem (scale the lengths)  
        [s1, s2, s3] = scaleFromStandard(lambda, sc_s1, sc_s2, sc_s3);

        % Construct the Dubins curve object with the computed optimal parameters
        curve = dubinscurve(x0, y0, th0, s1, s2, s3, ksigns(pidx,1)*Kmax, ksigns(pidx,2)*Kmax, ksigns(pidx,3)*Kmax);

        % Check the correctness of the algorithm
        assert(check(sc_s1, ksigns(pidx,1)*sc_Kmax, sc_s2, ksigns(pidx,2)*sc_Kmax, sc_s3, ksigns(pidx,3)*sc_Kmax, sc_th0, sc_thf));
      end
      k = ksigns(pidx,:)*sc_Kmax;
      lengths = [curve.a1.L  curve.a2.L  curve.a3.L];
    end


    
end
end


 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % Data structures
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Create a structure representing an arc of a Dubins curve (straight or
    % circular)
    function c = dubinsarc(x0, y0, th0, k, L)
      c.x0 = x0;
      c.y0 = y0;
      c.th0 = th0;
      c.k = k;
      c.L = L;
      [c.xf, c.yf, c.thf] = circline(L, x0, y0, th0, k);
    end

    % Create a structure representing a Dubins curve (composed by three arcs)
    function d = dubinscurve(x0, y0, th0, s1, s2, s3, k0, k1, k2)
      d = struct();
      d.a1 = dubinsarc(x0, y0, th0, k0, s1);
      d.a2 = dubinsarc(d.a1.xf, d.a1.yf, d.a1.thf, k1, s2);
      d.a3 = dubinsarc(d.a2.xf, d.a2.yf, d.a2.thf, k2, s3);
      d.L = d.a1.L + d.a2.L + d.a3.L;
    end



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % Auxiliary utility functions
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Implementation of function sinc(t), returning 1 for t==0, and sin(t)/t
    % otherwise
    function s = sinc(t)
      if (abs(t) < 0.002)
        % For small values of t use Taylor series approximation
        s = 1 - t^2/6 * (1 - t^2/20);
      else
        s = sin(t)/t;
      end
    end

    % Normalize an angle (in range [0,2*pi))
    function out = mod2pi(ang)
      out = ang;
      while (out < 0)
        out = out + 2 * pi;
      end
      while (out >= 2 * pi)
        out = out - 2 * pi;
      end
    end

    % Normalize an angular difference (range (-pi, pi])
    function out = rangeSymm(ang)
      out = ang;
      while (out <= -pi)
        out = out + 2 * pi;
      end
      while (out > pi)
        out = out - 2 * pi;
      end
    end

    % Check validity of a solution by evaluating explicitly the 3 equations 
    % defining a Dubins problem (in standard form)
    function bool = check(s1, k0, s2, k1, s3, k2, th0, thf)
      x0 = -1;
      y0 = 0;
      xf = 1;
      yf = 0;

      eq1 = x0 + s1 * sinc((1/2.) * k0 * s1) * cos(th0 + (1/2.) * k0 * s1) ...
               + s2 * sinc((1/2.) * k1 * s2) * cos(th0 + k0 * s1 + (1/2.) * k1 * s2) ...
               + s3 * sinc((1/2.) * k2 * s3) * cos(th0 + k0 * s1 + k1 * s2 + (1/2.) * k2 * s3) - xf;
      eq2 = y0 + s1 * sinc((1/2.) * k0 * s1) * sin(th0 + (1/2.) * k0 * s1) ...
               + s2 * sinc((1/2.) * k1 * s2) * sin(th0 + k0 * s1 + (1/2.) * k1 * s2) ...
               + s3 * sinc((1/2.) * k2 * s3) * sin(th0 + k0 * s1 + k1 * s2 + (1/2.) * k2 * s3) - yf;
      eq3 = rangeSymm(k0 * s1 + k1 * s2 + k2 * s3 + th0 - thf);

      Lpos = (s1 > 0) || (s2 > 0) || (s3 > 0);
      bool = (sqrt(eq1 * eq1 + eq2 * eq2 + eq3 * eq3) < 1.e-10) && Lpos;
    end



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % Functions to scale and solve Dubins problems
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Scale the input problem to standard form (x0: -1, y0: 0, xf: 1, yf: 0)
    function [sc_th0, sc_thf, sc_Kmax, lambda] = scaleToStandard(x0, y0, th0, xf, yf, thf, Kmax)
      % find transform parameters
      dx = xf - x0;
      dy = yf - y0;
      phi = atan2(dy, dx);
      lambda = hypot(dx, dy)/2;

      % scale and normalize angles and curvature
      sc_th0 = mod2pi(th0 - phi);
      sc_thf = mod2pi(thf - phi);
      sc_Kmax = Kmax * lambda;
    end

    % Scale the solution to the standard problem back to the original problem
    function [s1, s2, s3] = scaleFromStandard(lambda, sc_s1, sc_s2, sc_s3)
      s1 = sc_s1 * lambda;
      s2 = sc_s2 * lambda;
      s3 = sc_s3 * lambda;
    end

    % LSL
    function [ok, sc_s1, sc_s2, sc_s3] = LSL(sc_th0, sc_thf, sc_Kmax)
      invK = 1 / sc_Kmax;
      C = cos(sc_thf) - cos(sc_th0);
      S = 2 * sc_Kmax + sin(sc_th0) - sin(sc_thf);
      temp1 = atan2(C, S);
      sc_s1 = invK * mod2pi(temp1 - sc_th0);
      temp2 = 2 + 4 * sc_Kmax^2 - 2 * cos(sc_th0 - sc_thf) + 4 * sc_Kmax * (sin(sc_th0) - sin(sc_thf));
      if (temp2 < 0)
        ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
        return;
      end
      sc_s2 = invK * sqrt(temp2);
      sc_s3 = invK * mod2pi(sc_thf - temp1);
      ok = true;
    end

    % RSR
    function [ok, sc_s1, sc_s2, sc_s3] = RSR(sc_th0, sc_thf, sc_Kmax)
      invK = 1 / sc_Kmax;
      C = cos(sc_th0) - cos(sc_thf);
      S = 2 * sc_Kmax - sin(sc_th0) + sin(sc_thf);
      temp1 = atan2(C, S);
      sc_s1 = invK * mod2pi(sc_th0 - temp1);
      temp2 = 2 + 4 * sc_Kmax^2 - 2 * cos(sc_th0 - sc_thf) - 4 * sc_Kmax * (sin(sc_th0) - sin(sc_thf));
      if (temp2 < 0)
        ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
        return;
      end
      sc_s2 = invK * sqrt(temp2);
      sc_s3 = invK * mod2pi(temp1 - sc_thf);
      ok = true;
    end

    % LSR
    function [ok, sc_s1, sc_s2, sc_s3] = LSR(sc_th0, sc_thf, sc_Kmax)
      invK = 1 / sc_Kmax;
      C = cos(sc_th0) + cos(sc_thf);
      S = 2 * sc_Kmax + sin(sc_th0) + sin(sc_thf);
      temp1 = atan2(-C, S);
      temp3 = 4 * sc_Kmax^2 - 2 + 2 * cos(sc_th0 - sc_thf) + 4 * sc_Kmax * (sin(sc_th0) + sin(sc_thf));
      if (temp3 < 0)
        ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
        return;
      end
      sc_s2 = invK * sqrt(temp3);
      temp2 = -atan2(-2, sc_s2 * sc_Kmax);
      sc_s1 = invK * mod2pi(temp1 + temp2 - sc_th0);
      sc_s3 = invK * mod2pi(temp1 + temp2 - sc_thf);
      ok = true;
    end

    % RSL
    function [ok, sc_s1, sc_s2, sc_s3] = RSL(sc_th0, sc_thf, sc_Kmax)
      invK = 1 / sc_Kmax;
      C = cos(sc_th0) + cos(sc_thf);
      S = 2 * sc_Kmax - sin(sc_th0) - sin(sc_thf);
      temp1 = atan2(C, S);
      temp3 = 4 * sc_Kmax^2 - 2 + 2 * cos(sc_th0 - sc_thf) - 4 * sc_Kmax * (sin(sc_th0) + sin(sc_thf));
      if (temp3 < 0)
        ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
        return;
      end
      sc_s2 = invK * sqrt(temp3);
      temp2 = atan2(2, sc_s2 * sc_Kmax);
      sc_s1 = invK * mod2pi(sc_th0 - temp1 + temp2);
      sc_s3 = invK * mod2pi(sc_thf - temp1 + temp2);
      ok = true;
    end

    % RLR
    function [ok, sc_s1, sc_s2, sc_s3] = RLR(sc_th0, sc_thf, sc_Kmax)
      invK = 1 / sc_Kmax;
      C = cos(sc_th0) - cos(sc_thf);
      S = 2 * sc_Kmax - sin(sc_th0) + sin(sc_thf);
      temp1 = atan2(C, S);
      temp2 = 0.125 * (6 - 4 * sc_Kmax^2 + 2 * cos(sc_th0 - sc_thf) + 4 * sc_Kmax * (sin(sc_th0) - sin(sc_thf)));
      if (abs(temp2) > 1)
        ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
        return;
      end
      sc_s2 = invK * mod2pi(2 * pi - acos(temp2));
      sc_s1 = invK * mod2pi(sc_th0 - temp1 + 0.5 * sc_s2 * sc_Kmax);
      sc_s3 = invK * mod2pi(sc_th0 - sc_thf + sc_Kmax * (sc_s2 - sc_s1));
      ok = true;
    end

    % LRL
    function [ok, sc_s1, sc_s2, sc_s3] = LRL(sc_th0, sc_thf, sc_Kmax)
      invK = 1 / sc_Kmax;
      C = cos(sc_thf) - cos(sc_th0);
      S = 2 * sc_Kmax + sin(sc_th0) - sin(sc_thf);
      temp1 = atan2(C, S);
      temp2 = 0.125 * (6 - 4 * sc_Kmax^2 + 2 * cos(sc_th0 - sc_thf) - 4 * sc_Kmax * (sin(sc_th0) - sin(sc_thf)));
      if (abs(temp2) > 1)
        ok = false; sc_s1 = 0; sc_s2 = 0; sc_s3 = 0;
        return;
      end
      sc_s2 = invK * mod2pi(2 * pi - acos(temp2));
      sc_s1 = invK * mod2pi(temp1 - sc_th0 + 0.5 * sc_s2 * sc_Kmax);
      sc_s3 = invK * mod2pi(sc_thf - sc_th0 + sc_Kmax * (sc_s2 - sc_s1));
      ok = true;
    end

    % Solve the Dubins problem for the given input parameters.
    % Return the type and the parameters of the optimal curve
   


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % Functions to plot Dubins curves
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Evaluate an arc (circular or straight) composing a Dubins curve, at a 
    % given arc-length s
    function [x, y, th] = circline(s, x0, y0, th0, k)
      x = x0 + s * sinc(k * s / 2.0) * cos(th0 + k * s / 2);
      y = y0 + s * sinc(k * s / 2.0) * sin(th0 + k * s / 2);
      th = mod2pi(th0 + k * s);
    end


  

