function new_string = my_zfill( string,len)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
string_len = length(string);
new_string =string;
if (string_len>len)&&(string_len==len)
    disp('the string is not longer that the number you specific');
else
    for i=1:(len-string_len)
        new_string=['0',new_string];
    end
end
end

