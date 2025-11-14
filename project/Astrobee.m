% ASTROBEE CLASS DEFINITION
% VERSION 3.0
% POC: SAMI HAQ & ADARSH RAJGURU

classdef Astrobee
    properties
        Position % 3x1 vector
        Orientation % 3x1 unit vector (current direction)
        errorMagnitude % Scalar float value
    end
    
    methods
        function obj = Astrobee(position, orientation, errorMagnitude)
            obj.Position = position;
            obj.Orientation = orientation / norm(orientation); % Ensure unit vector
            obj.errorMagnitude = errorMagnitude;
        end
        
        function obj = rotate(obj, quaternion)
            % Rotate orientation by quaternion
            q = quaternion; % [q0, qx, qy, qz]
            R = quat2rotm(q); % Convert quaternion to rotation matrix
            obj.Orientation = R * obj.Orientation; % Rotate orientation
        end
        
        function obj = move(obj, distance)
            randomError = (rand(3, 1) - 0.5) * 2 * obj.errorMagnitude; % Random error vector
            %movement = distance * obj.Orientation; % Movement with no error
            movement = distance * obj.Orientation + randomError; % Add error to movement
            obj.Position = obj.Position + movement;
        end
    end
end
