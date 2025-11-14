clear; clc
close all

## PRESETS

# -----------------------------
# geometry
# -----------------------------

# arm length
d = 0.2; # [m]

# propeller radius
r = 0.05; # [m]


# -----------------------------
# dynamic & inertial
# -----------------------------

# body mass
m = 0.5; # [kg]

# principal moments of inertia 
Ixxb = 0.01; # [kg m^2]
Iyyb = 0.01; # [kg m^2]
Izzb = 0.05; # [kg m^2]

# propeller force (constant)
f = 0.25; # [N]


# -----------------------------
# initial conditions
# -----------------------------

# position
p0 = [0.5; 0.5; 1]; # m

# velocity
p_dot0 = [0.1; 0.05; 0.1]; # m/s

# ZYX euler angles, specify as [yaw, pitch, roll]
euler0 = [pi/8, 0, pi/4]; # [rad]

# angular velocity
wb0 = [0.2; 0.1; 1]; # [rad/s]


# -----------------------------
# sim settings
# -----------------------------

# sim time
tmax = 2; # [s]

# printing (boolean)
print = true;

# extra plots (boolean)
extra_plots = true;







## SIMULATION CODE [TRY NOT TO BREAK]

function S = skew(v)
    # v must be a 3x1 vector
    S = [  0   -v(3)  v(2);
          v(3)   0   -v(1);
         -v(2)  v(1)   0 ];
end

function dxdt = odefun(t, x, rb, Fb, Ib, m, g)
    
    xp_dot = x(13:15);
    R = reshape(x(4:12), [3, 3]);
    wb = x(16:end);

    R_dot = R * skew(wb);
    Rbar_dot = reshape(R_dot, [], 1);

    Fb1 = Fb(:, 1);
    Fb2 = Fb(:, 2);
    Fb3 = Fb(:, 3);
    Fb4 = Fb(:, 4);

    wb_dot = Ib \ ( ...
        cross(rb(:, 1), Fb1) + ...
        cross(rb(:, 2), Fb2) + ...
        cross(rb(:, 3), Fb3) + ...
        cross(rb(:, 4), Fb4) - ...
        cross(wb, Ib * wb) );

    F = R * (Fb1 + Fb2 + Fb3 + Fb4);
    

    dxdt = [
        xp_dot;
        Rbar_dot;
        F/m + g;
        wb_dot
    ];

end

Rbar0 = reshape(eul2rotm(euler0, 'ZYX'), [], 1); # rad

x0 = [p0; Rbar0; p_dot0; wb0];

r1 = [d; 0; 0];
r2 = [0; d; 0];
r3 = [-d; 0; 0];
r4 = [0; -d; 0];

F1 = f*[0; 0; 1];
F2 = f*[0; 0; 1];
F3 = f*[0; 0; 1];
F4 = f*[0; 0; 1];

rb = [r1, r2, r3, r4]; # m
Fb = [F1, F2, F3, F4]; # N


Ib = diag([Ixxb, Iyyb, Izzb]); # kg m^2

g = 9.81*[0; 0; -1]; # m/s^2

tspan = [0, tmax];
[t, x] = ode45(@odefun, tspan, x0, [], rb, Fb, Ib, m, g);


## ---------- plot the system ----------

xs = x(:, 1);
ys = x(:, 2);
zs = x(:, 3);

Rbar_f = reshape(x(:, 4:12).', 3, 3, []);
eul_f = rotm2eul(Rbar_f, 'ZYX');
phis = eul_f(:, 3);
thetas = eul_f(:, 2);
psis = eul_f(:, 1);

if extra_plots

    subplot(3, 1, 1)
    plot(t, xs)
    xlabel("$t$ [s]", Interpreter="latex")
    ylabel("$x_{b}(t)$", Interpreter="latex")
    grid on
    
    subplot(3, 1, 2)
    plot(t, ys)
    xlabel("$t$ [s]", Interpreter="latex")
    ylabel("$y_{b}(t)$", Interpreter="latex")
    grid on
    
    subplot(3, 1, 3)
    plot(t, zs)
    xlabel("$t$ [s]", Interpreter="latex")
    ylabel("$z_{b}(t)$", Interpreter="latex")
    grid on
    
    
    figure;
    
    subplot(3, 1, 1)
    plot(t, phis)
    xlabel("$t$ [s]", 'Interpreter','latex')
    ylabel("$\phi(t)$, roll [rad]", 'Interpreter','latex')
    grid on
    
    subplot(3, 1, 2)
    plot(t, thetas)
    xlabel("$t$ [s]", Interpreter="latex")
    ylabel("$\theta(t)$, pitch [rad]", Interpreter="latex")
    grid on
    
    subplot(3, 1, 3)
    plot(t, psis)
    xlabel("$t$ [s]", Interpreter="latex")
    ylabel("$\psi(t)$, yaw [rad]", Interpreter="latex")
    grid on

end


## ---------- animate the system for 2 seconds ----------

figure;
delete("drone.gif");

function prop_coords = getProps(C, u, v, r)

    # C := center of circle
    
    # Parameterize circle
    theta = linspace(0, 2*pi, 400);
    x_prop = C(1) + r*cos(theta)*u(1) + r*sin(theta)*v(1);
    y_prop = C(2) + r*cos(theta)*u(2) + r*sin(theta)*v(2);
    z_prop = C(3) + r*cos(theta)*u(3) + r*sin(theta)*v(3);

    prop_coords = [x_prop; y_prop; z_prop]; # return as 3-by-N
end

for i = 1:length(t)

    x_cm = xs(i);
    y_cm = ys(i);
    z_cm = zs(i);
    com = [x_cm; y_cm; z_cm];

    phi = phis(i);
    theta = thetas(i);
    psi = psis(i);

    Ri = Rbar_f(:, :, i);
    
    # define body frame vectors for each arm
    # apply rotation Ri to transform each body frame vector into the world frame
    # add com vector (which is in the world frame) to get EE position of each arm in the global grame
    green_arm =  com + Ri * [0; d; 0]; # green arm
    red_arm   =  com + Ri * [d; 0; 0]; # red arm
    blue_arm  =  com + Ri * [-d; 0; 0]; # blue arm
    black_arm =  com + Ri * [0; -d; 0]; # black arm

    plot3([com(1) green_arm(1)], [com(2) green_arm(2)], [com(3) green_arm(3)], Color="g")
    hold on
    plot3([com(1) red_arm(1)], [com(2) red_arm(2)], [com(3) red_arm(3)], Color="r")
    plot3([com(1) blue_arm(1)], [com(2) blue_arm(2)], [com(3) blue_arm(3)], Color="b") 
    plot3([com(1) black_arm(1)], [com(2) black_arm(2)], [com(3) black_arm(3)], Color="k") 

    # define orthonormal basis for plotting propellers
    u = (green_arm - com)/d;
    v = (red_arm - com)/d;
    
    props = getProps(green_arm, u, v, r);
    plot3(props(1,:), props(2,:), props(3,:), Color="k") # green prop
    
    props = getProps(red_arm, u, v, r);
    plot3(props(1,:), props(2,:), props(3,:), Color="k") # red prop
    
    props = getProps(blue_arm, u, v, r);
    plot3(props(1,:), props(2,:), props(3,:), Color="k") # blue prop
    
    props = getProps(black_arm, u, v, r);
    plot3(props(1,:), props(2,:), props(3,:), Color="k") # black prop

    xlabel("$x_{COM}(t)$ [m]", Interpreter="latex")
    ylabel("$y_{COM}(t)$ [m]", Interpreter="latex")
    zlabel("$z_{COM}(t)$ [m]", Interpreter="latex")
    grid on


    xlim([min(xs)-d-r, max(xs)+d+r])
    ylim([min(ys)-d-r, max(ys)+d+r])
    zlim([min(zs)-d-r, max(zs)+d+r])


    if print
        fprintf('t = #.3f s, x_cm = #.3f m, y_cm = #.3f m, z_cm = #.3f m\n', t(i), x_cm, y_cm, z_cm);
    end


    exportgraphics(gcf, "drone" + ".gif", 'Append',true);

    clf;

end