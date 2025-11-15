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