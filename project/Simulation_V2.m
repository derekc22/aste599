
% ASTROBEE-CLINGERS DOCKING SIM — RECORDED VERSION
% VERSION 3.5 + Video Capture
% POC: SAMI HAQ & ADARSH RAJGURU
% Mode 1: Two dynamic astrobees translate and rotate
% Mode 2: Target can rotate but cannot translate
% Mode 3: Target cannot rotate and translate
%
% Notes:
% - Requires an Astrobee class with:
%     - fields: Position (3x1), Orientation (3x1)
%     - constructor: Astrobee(position, orientation, movementError)
%     - method: move(distance) -> returns updated Astrobee
% - This script records the animation to an MP4 using VideoWriter.
% - The video is saved in the current working directory.
%
% ---------------------------------------------------------------

clear all; close all; clc; 

% -----------------------------
% User Inputs:
% -----------------------------
prompt1 = "Input Astrobee Step Size (in meters): ";
prompt2 = "Input Astrobee Movement Error (in meters): ";
prompt3 = "Which mode would you like to run? Please enter 1, 2, or 3 to define the modes: ";

stepDistance = input(prompt1);
movementError = input(prompt2);
mode = input(prompt3);

% -----------------------------
% Simulation parameters
% -----------------------------
cubeSize   = 3;      % 3x3x3 cube (meters)
minDistance = 0.1;   % Center-to-center stopping distance (meters)
waitTime   = 0.3;    % Pause duration to visualize movement
rng('shuffle');      % Random seed

% -----------------------------
% Initialize positions with >= 2 m separation
% -----------------------------
while true
    position1 = rand(3, 1) * (cubeSize);
    position2 = rand(3, 1) * (cubeSize);
    if norm(position1 - position2) >= 2
        break;
    end
end

% Random initial orientations (normalized)
orientation1 = rand(3, 1) - 0.5; orientation1 = orientation1 / norm(orientation1);
orientation2 = rand(3, 1) - 0.5; orientation2 = orientation2 / norm(orientation2);

% Create Astrobee objects
astrobee1 = Astrobee(position1, orientation1, movementError);
astrobee2 = Astrobee(position2, orientation2, movementError);

% Save trajectories
trajectory1 = astrobee1.Position';
trajectory2 = astrobee2.Position';

% Sphere mesh for the Astrobees
[sphereX, sphereY, sphereZ] = sphere();
sphereRadius = 0.03; % meters

% -----------------------------
% Figure and axes
% -----------------------------
f = figure('Name','Astrobee CLINGERS Docking','Color','w');
set(f, 'WindowState', 'maximized');
set(f, 'Renderer', 'opengl');
hold on; grid on; axis equal;
xlim([0, cubeSize]); ylim([0, cubeSize]); zlim([0, cubeSize]);
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('3D Animated Trajectories and Orientations of Astrobees');
view(3);

% -----------------------------
% VIDEO CAPTURE SETUP
% -----------------------------
videoFileName = sprintf('AstrobeeDocking_Mode%d.mp4', mode);
v = VideoWriter(videoFileName, 'MPEG-4');
v.FrameRate = 10;     % Increase for smoother video (larger file size)
open(v);

% -----------------------------
% Plot elements
% -----------------------------
plot1 = plot3(NaN, NaN, NaN, 'r-', 'LineWidth', 1.5); % Trajectory 1
plot2 = plot3(NaN, NaN, NaN, 'b-', 'LineWidth', 1.5); % Trajectory 2

astrobee1Sphere = surf(sphereRadius * sphereX + astrobee1.Position(1), ...
                       sphereRadius * sphereY + astrobee1.Position(2), ...
                       sphereRadius * sphereZ + astrobee1.Position(3), ...
                       'FaceColor','red','EdgeColor','none','FaceAlpha',0.7);
astrobee2Sphere = surf(sphereRadius * sphereX + astrobee2.Position(1), ...
                       sphereRadius * sphereY + astrobee2.Position(2), ...
                       sphereRadius * sphereZ + astrobee2.Position(3), ...
                       'FaceColor','blue','EdgeColor','none','FaceAlpha',0.7);

% Initial orientations (static quivers)
initialOrientation1 = quiver3(astrobee1.Position(1), astrobee1.Position(2), astrobee1.Position(3), ...
    orientation1(1), orientation1(2), orientation1(3), ...
    'r-.','LineWidth',1.5,'MaxHeadSize',0.5,'DisplayName','Initial Orientation 1');
initialOrientation2 = quiver3(astrobee2.Position(1), astrobee2.Position(2), astrobee2.Position(3), ...
    orientation2(1), orientation2(2), orientation2(3), ...
    'b-.','LineWidth',1.5,'MaxHeadSize',0.5,'DisplayName','Initial Orientation 2');

% Current orientations (dynamic quivers)
orientation1Plot = quiver3(NaN, NaN, NaN, NaN, NaN, NaN, ...
    'r','LineWidth',2,'MaxHeadSize',0.5,'DisplayName','Current Orientation 1');
orientation2Plot = quiver3(NaN, NaN, NaN, NaN, NaN, NaN, ...
    'b','LineWidth',2,'MaxHeadSize',0.5,'DisplayName','Current Orientation 2');

% Legend placeholders (distance + mode)
distancePoint = plot3(NaN, NaN, NaN, '-.', 'MarkerFaceColor','none','MarkerEdgeColor','none', ...
                      'DisplayName', sprintf('Distance: %.3f m', 0));
modePoint     = plot3(NaN, NaN, NaN, 'diamond', 'MarkerFaceColor','none','MarkerEdgeColor','r', ...
                      'DisplayName', sprintf('Mode: %d', mode));

legendHandle = legend([modePoint, plot1, plot2, initialOrientation1, initialOrientation2, distancePoint], ...
    '', 'Astrobee 1 Path', 'Astrobee 2 Path', 'Astrobee 1 Initial Orientation', 'Astrobee 2 Initial Orientation', '', ...
    'FontSize', 14, 'Location', 'northeastoutside');

% Capture initial frame
drawnow;
frame = getframe(f);
writeVideo(v, frame);

% -----------------------------
% Main Simulation Loop
% -----------------------------
try
    while true
        % Compute distance
        distanceBetween = norm(astrobee1.Position - astrobee2.Position);

        % Update legend strings
        set(distancePoint, 'DisplayName', sprintf('Distance: %.3f m', distanceBetween));
        set(modePoint,     'DisplayName', sprintf('Mode: %d', mode));
        legendHandle.String = get(legendHandle, 'String'); % refresh

        % Break if within stopping distance
        if distanceBetween <= minDistance
            break;
        end

        % ---- Step 1: Move along current orientations ----
        moveDistance = min(stepDistance, (distanceBetween - minDistance) / 2);
        astrobee1 = astrobee1.move(moveDistance);
        if mode == 1
            astrobee2 = astrobee2.move(moveDistance);
        end

        % Update trajectories
        trajectory1 = [trajectory1; astrobee1.Position'];
        trajectory2 = [trajectory2; astrobee2.Position'];

        % Update trajectory plots
        set(plot1, 'XData', trajectory1(:,1), 'YData', trajectory1(:,2), 'ZData', trajectory1(:,3));
        set(plot2, 'XData', trajectory2(:,1), 'YData', trajectory2(:,2), 'ZData', trajectory2(:,3));

        % Update Astrobee spheres
        set(astrobee1Sphere, 'XData', sphereRadius*sphereX + astrobee1.Position(1), ...
                             'YData', sphereRadius*sphereY + astrobee1.Position(2), ...
                             'ZData', sphereRadius*sphereZ + astrobee1.Position(3));
        set(astrobee2Sphere, 'XData', sphereRadius*sphereX + astrobee2.Position(1), ...
                             'YData', sphereRadius*sphereY + astrobee2.Position(2), ...
                             'ZData', sphereRadius*sphereZ + astrobee2.Position(3));

        % Keep previous orientation during movement (draw at new positions)
        set(orientation1Plot, 'XData', astrobee1.Position(1), 'YData', astrobee1.Position(2), 'ZData', astrobee1.Position(3), ...
                              'UData', astrobee1.Orientation(1), 'VData', astrobee1.Orientation(2), 'WData', astrobee1.Orientation(3));
        set(orientation2Plot, 'XData', astrobee2.Position(1), 'YData', astrobee2.Position(2), 'ZData', astrobee2.Position(3), ...
                              'UData', astrobee2.Orientation(1), 'VData', astrobee2.Orientation(2), 'WData', astrobee2.Orientation(3));

        drawnow;
        pause(waitTime);

        % Record frame after movement
        frame = getframe(f);
        writeVideo(v, frame);

        % ---- Step 2: Reorient to face each other ----
        direction1To2 = (astrobee2.Position - astrobee1.Position) / norm(astrobee2.Position - astrobee1.Position);
        astrobee1.Orientation = direction1To2;

        if mode == 1 || mode == 2
            direction2To1 = (astrobee1.Position - astrobee2.Position) / norm(astrobee1.Position - astrobee2.Position);
            astrobee2.Orientation = direction2To1;
        end

        % Update orientation quivers after reorientation
        set(orientation1Plot, 'UData', astrobee1.Orientation(1), 'VData', astrobee1.Orientation(2), 'WData', astrobee1.Orientation(3));
        set(orientation2Plot, 'UData', astrobee2.Orientation(1), 'VData', astrobee2.Orientation(2), 'WData', astrobee2.Orientation(3));

        drawnow;
        pause(waitTime);

        % Record frame after reorientation
        frame = getframe(f);
        writeVideo(v, frame);
    end
catch ME
    warning('Simulation stopped due to error: %s', ME.message);
end

% Final hold state
hold off;

% -----------------------------
% Finalize video
% -----------------------------
try
    % Capture the final frame one last time
    drawnow;
    frame = getframe(f);
    writeVideo(v, frame);
catch
    % If capturing fails at the end, continue closing the file
end

% Ensure the video writer is closed
try
    close(v);
    fprintf('Video saved as: %s\n', videoFileName);
catch ME
    warning('Failed to properly close video: %s', ME.message);
end
