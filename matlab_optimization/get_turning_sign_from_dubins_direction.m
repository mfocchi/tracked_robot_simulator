function s = get_turning_sign_from_dubins_direction(dir)
    if(strcmp("L", dir))
        s = 1;
    elseif(strcmp("R", dir))
        s = -1;
    elseif(strcmp("S", dir))
        s = 0;
    else
        disp("ERROR: No direction has been defined!")
        s = [];
    end
end
