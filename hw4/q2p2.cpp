#include <vector>
#include <string>
#include <iostream>

using std::vector; using std::string; using std::cout; using std::endl; using std::size_t;

struct State {
    double x, z, w, vx, vz, eps; // w = omega, eps = angular rate
};

struct Action {
    double T;   // thrust (body frame, upward)
    double tau; // torque
};

struct Limits6 {
    // symmetric limits about zero for each state dimension
    double x, z, w, vx, vz, eps;
};

class QuadDynamics {
public:
    double m{1.0};
    double I{0.01};
    double g{9.81};
    double dt{0.05};

    QuadDynamics() = default;
    QuadDynamics(double m_, double I_, double g_, double dt_) : m(m_), I(I_), g(g_), dt(dt_) {}

    inline State step(const State &s, const Action &a) const {
        // Forward Euler discretization
        State sn;
        sn.x   = s.x   + dt * s.vx;
        sn.z   = s.z   + dt * s.vz;
        sn.w   = s.w   + dt * s.eps;         // w' = w + dt * eps
        sn.vx  = s.vx  + dt * ( (a.T / m) * std::sin(s.w) );
        sn.vz  = s.vz  + dt * ( (a.T / m) * std::cos(s.w) - g );
        sn.eps = s.eps + dt * ( a.tau / I );
        return sn;
    }
};

// 6D rectilinear grid, symmetric bounds around zero
class Grid6D {
public:
    // grid axes: x, z, w, vx, vz, eps
    vector<double> axes[6];
    double minv[6];
    double maxv[6];
    size_t dims[6];

    Grid6D(const Limits6 &smax, const vector<double> &h) {
        // h size must be 6
        assert(h.size() == 6);
        const double mins[6] = {-smax.x, -smax.z, -smax.w, -smax.vx, -smax.vz, -smax.eps};
        const double maxs[6] = { smax.x,  smax.z,  smax.w,  smax.vx,  smax.vz,  smax.eps};
        for (int k = 0; k < 6; ++k) {
            minv[k] = mins[k];
            maxv[k] = maxs[k];
            axes[k] = linspace(minv[k], maxv[k], h[k]);
            dims[k] = axes[k].size();
        }
    }

    static vector<double> linspace(double a, double b, double step) {
        // inclusive of both ends
        int n = static_cast<int>(std::floor((b - a) / step + 0.5)) + 1;
        vector<double> v; v.reserve(std::max(2, n));
        double x = a;
        for (int i = 0; i < n - 1; ++i) { v.push_back(x); x += step; }
        v.push_back(b);
        return v;
    }

    size_t size() const {
        return dims[0]*dims[1]*dims[2]*dims[3]*dims[4]*dims[5];
    }

    inline size_t flatten(const std::array<size_t,6> &idx) const {
        // row-major
        size_t stride = 1, flat = 0;
        for (int k = 0; k < 6; ++k) {
            flat += idx[k] * stride;
            stride *= dims[k];
        }
        return flat;
    }

    inline std::array<size_t,6> clampIndex(const std::array<long long,6> &idxLL) const {
        std::array<size_t,6> idx{};
        for (int k = 0; k < 6; ++k) {
            long long v = std::max(0LL, std::min<long long>(static_cast<long long>(dims[k]-1), idxLL[k]));
            idx[k] = static_cast<size_t>(v);
        }
        return idx;
    }

    inline void projectWithin(State &s) const {
        double *ptr[6] = { &s.x, &s.z, &s.w, &s.vx, &s.vz, &s.eps };
        for (int k = 0; k < 6; ++k) {
            if (*ptr[k] < minv[k]) *ptr[k] = minv[k];
            if (*ptr[k] > maxv[k]) *ptr[k] = maxv[k];
        }
    }

    // find hypercube around s and interpolation weights
    struct Cell {
        std::array<size_t,6> i0; // lower corner indices
        double t[6];             // fractional coordinates in [0,1]
    };

    Cell locate(const State &s_in) const {
        Cell cell{};
        const double vals[6] = {s_in.x, s_in.z, s_in.w, s_in.vx, s_in.vz, s_in.eps};
        for (int k = 0; k < 6; ++k) {
            // find index i such that axes[k][i] <= v <= axes[k][i+1]
            const vector<double> &ax = axes[k];
            if (vals[k] <= ax.front()) { cell.i0[k] = 0; cell.t[k] = 0.0; continue; }
            if (vals[k] >= ax.back())  { cell.i0[k] = ax.size()-2; cell.t[k] = 1.0; continue; }
            auto it = std::upper_bound(ax.begin(), ax.end(), vals[k]);
            size_t i1 = std::distance(ax.begin(), it);
            size_t i0 = i1 - 1;
            double a0 = ax[i0], a1 = ax[i1];
            cell.i0[k] = i0;
            cell.t[k]  = (vals[k] - a0) / std::max(1e-12, (a1 - a0));
        }
        return cell;
    }

    // 6D multilinear interpolation over 64 corners
    double interp(const vector<double> &V, const State &s) const {
        Cell c = locate(s);
        double acc = 0.0;
        for (int mask = 0; mask < 64; ++mask) {
            double w = 1.0; std::array<size_t,6> idx{};
            for (int k = 0; k < 6; ++k) {
                size_t i = c.i0[k] + ((mask >> k) & 1);
                if (i >= dims[k]) { w = 0.0; break; }
                idx[k] = i;
                double tk = c.t[k];
                w *= ((mask >> k) & 1) ? tk : (1.0 - tk);
            }
            if (w == 0.0) continue;
            acc += w * V[flatten(idx)];
        }
        return acc;
    }
};

class ActionGrid {
public:
    vector<Action> actions;

    ActionGrid(double T_max, double tau_max, double hT, double htau) {
        vector<double> Ts = Grid6D::linspace(-T_max, T_max, hT);
        vector<double> taus = Grid6D::linspace(-tau_max, tau_max, htau);
        actions.reserve(Ts.size()*taus.size());
        for (double T : Ts) for (double t : taus) actions.push_back({T, t});
    }

    size_t size() const { return actions.size(); }
    const Action &operator[](size_t i) const { return actions[i]; }
};

class ValueIteration6D {
public:
    const Grid6D &grid;
    const ActionGrid &A;
    QuadDynamics dyn;
    double gamma{0.99};
    vector<double> V, V_new;

    ValueIteration6D(const Grid6D &g, const ActionGrid &ag, const QuadDynamics &d, double gamma_)
        : grid(g), A(ag), dyn(d), gamma(gamma_), V(grid.size(), 0.0), V_new(grid.size(), 0.0) {}

    static inline double running_cost(const State &s) {
        // Q = I, cost = ||s||^2
        return s.x*s.x + s.z*s.z + s.w*s.w + s.vx*s.vx + s.vz*s.vz + s.eps*s.eps;
    }

    inline State state_at(const std::array<size_t,6> &idx) const {
        return State{ grid.axes[0][idx[0]], grid.axes[1][idx[1]], grid.axes[2][idx[2]],
                      grid.axes[3][idx[3]], grid.axes[4][idx[4]], grid.axes[5][idx[5]] };
    }

    double iterate_once() {
        double max_delta = 0.0;
        // loop over all grid states
        std::array<size_t,6> idx{};
        for (idx[0]=0; idx[0]<grid.dims[0]; ++idx[0])
        for (idx[1]=0; idx[1]<grid.dims[1]; ++idx[1])
        for (idx[2]=0; idx[2]<grid.dims[2]; ++idx[2])
        for (idx[3]=0; idx[3]<grid.dims[3]; ++idx[3])
        for (idx[4]=0; idx[4]<grid.dims[4]; ++idx[4])
        for (idx[5]=0; idx[5]<grid.dims[5]; ++idx[5]) {
            size_t f = grid.flatten(idx);
            State s = state_at(idx);
            double best = std::numeric_limits<double>::infinity();
            // minimize cost-to-go
            for (const Action &a : A.actions) {
                State sn = dyn.step(s, a);
                // project to bounds for interpolation stability
                State snp = sn; grid.projectWithin(snp);
                double J = running_cost(s) + gamma * grid.interp(V, snp);
                if (J < best) best = J;
            }
            V_new[f] = best;
            max_delta = std::max(max_delta, std::fabs(V_new[f] - V[f]));
        }
        V.swap(V_new);
        return max_delta;
    }

    void run(size_t max_iters, double tol, bool verbose=true) {
        for (size_t it = 0; it < max_iters; ++it) {
            double d = iterate_once();
            if (verbose && (it % 1 == 0)) {
                cout << "Iter " << it << ": sup-norm delta = " << d << "\n";
            }
            if (d < tol) {
                cout << "Converged in " << it+1 << " iterations, tol = " << tol << "\n";
                return;
            }
        }
        cout << "Reached max iterations without meeting tolerance.\n";
    }

    Action greedy_action(const State &s) const {
        double best = std::numeric_limits<double>::infinity();
        Action bestA{0,0};
        for (const Action &a : A.actions) {
            State sn = dyn.step(s, a);
            State snp = sn; grid.projectWithin(snp);
            double J = running_cost(s) + gamma * grid.interp(V, snp);
            if (J < best) { best = J; bestA = a; }
        }
        return bestA;
    }

    // simulate greedy policy until near origin or steps exhausted
    vector<State> rollout(State s0, size_t max_steps, double tol_norm=1e-2) const {
        vector<State> traj; traj.reserve(max_steps+1); traj.push_back(s0);
        for (size_t k = 0; k < max_steps; ++k) {
            double nrm = std::sqrt(s0.x*s0.x + s0.z*s0.z + s0.w*s0.w + s0.vx*s0.vx + s0.vz*s0.vz + s0.eps*s0.eps);
            if (nrm < tol_norm) break;
            Action a = greedy_action(s0);
            State s1 = dyn.step(s0, a);
            State s1p = s1; grid.projectWithin(s1p);
            s0 = s1p;
            traj.push_back(s0);
        }
        return traj;
    }

    // Save a 2D slice V(x,z) fixing other states at zero
    void save_slice_xz(const string &fname) const {
        // find indices nearest to 0 for w, vx, vz, eps
        auto near_zero_idx = [&](int k){
            const auto &ax = grid.axes[k];
            size_t best = 0; double bd = std::fabs(ax[0]);
            for (size_t i=1;i<ax.size();++i){ double d = std::fabs(ax[i]); if (d < bd){ bd=d; best=i; } }
            return best;
        };
        size_t iw = near_zero_idx(2), ivx = near_zero_idx(3), ivz = near_zero_idx(4), ieps = near_zero_idx(5);
        std::ofstream f(fname);
        f << std::setprecision(8);
        for (size_t iz=0; iz<grid.dims[1]; ++iz) {
            for (size_t ix=0; ix<grid.dims[0]; ++ix) {
                std::array<size_t,6> idx{ix,iz,iw,ivx,ivz,ieps};
                double val = V[grid.flatten(idx)];
                f << val; if (ix + 1 < grid.dims[0]) f << ",";
            }
            f << "\n";
        }
        f.close();
    }

    // Save averaged V over w,vx,vz,eps producing Vbar(x,z)
    void save_avg_xz(const string &fname) const {
        std::ofstream f(fname);
        f << std::setprecision(8);
        for (size_t iz=0; iz<grid.dims[1]; ++iz) {
            for (size_t ix=0; ix<grid.dims[0]; ++ix) {
                long double acc = 0.0L; size_t cnt = 0;
                for (size_t iw=0; iw<grid.dims[2]; ++iw)
                for (size_t ivx=0; ivx<grid.dims[3]; ++ivx)
                for (size_t ivz=0; ivz<grid.dims[4]; ++ivz)
                for (size_t ieps=0; ieps<grid.dims[5]; ++ieps) {
                    std::array<size_t,6> idx{ix,iz,iw,ivx,ivz,ieps};
                    acc += V[grid.flatten(idx)];
                    ++cnt;
                }
                double val = static_cast<double>(acc / (long double)cnt);
                f << val; if (ix + 1 < grid.dims[0]) f << ",";
            }
            f << "\n";
        }
        f.close();
    }
};

static inline double to_double(const char* s, double def){ return s ? std::atof(s) : def; }

int main(int argc, char** argv) {
    // Defaults from the assignment
    Limits6 smax{0.5, 0.5, 0.5, 0.1, 0.1, 0.1};
    double umax_T = 1.0, umax_tau = 0.1;

    // Steps, may be overridden by CLI
    vector<double> h = {0.05, 0.05, 0.05, 0.05, 0.05, 0.05};
    double hT = 0.2, htau = 0.02;

    // Dynamics
    double dt = 0.05;
    QuadDynamics dyn(1.0, 0.01, 9.81, dt);

    // Value iteration params
    double gamma = 0.99; double tol = 1e-3; size_t max_iters = 200;

    // Optional overrides: gamma, tol, max_iters
    if (argc >= 2) gamma = to_double(argv[1], gamma);
    if (argc >= 3) tol   = to_double(argv[2], tol);
    if (argc >= 4) max_iters = static_cast<size_t>(std::atoll(argv[3]));

    cout << "Building grids...\n";
    Grid6D grid(smax, h);
    ActionGrid A(umax_T, umax_tau, hT, htau);
    cout << "State grid points: " << grid.size() << "\n";
    cout << "Action grid points: " << A.size() << "\n";

    ValueIteration6D VI(grid, A, dyn, gamma);

    auto t0 = std::chrono::high_resolution_clock::now();
    VI.run(max_iters, tol, true);
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    cout << "Total time: " << sec << " s\n";

    std::filesystem::create_directory("data");
    VI.save_slice_xz("data/V_slice_xz.csv");
    VI.save_avg_xz("data/V_avg_xz.csv");
    cout << "Saved V(x,z) slice to data/V_slice_xz.csv and averaged V to data/V_avg_xz.csv\n";

    // Policy rollout from the specified initial state
    State s0{0.4, 0.4, 0.2, 0.05, 0.05, 0.05};
    auto traj = VI.rollout(s0, 500, 1e-3);
    std::ofstream ft("data/trajectory.csv");
    ft << "x,z,w,vx,vz,eps\n";
    for (const auto &s : traj) {
        ft << s.x << "," << s.z << "," << s.w << "," << s.vx << "," << s.vz << "," << s.eps << "\n";
    }
    ft.close();
    cout << "Saved greedy-policy trajectory to data/trajectory.csv with " << traj.size() << " states\n";

    return 0;
}
