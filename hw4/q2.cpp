#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <fstream>

using std::string, std::cout, std::endl;



class Actor {

    public:

        double epsilon; // exploration fraction

        Actor(double epsilon) : epsilon(epsilon) {};

        Actor() : Actor(0.5) {};    
};

class GridWorld {

    public:

    using coord = std::pair<int, int>;

        int size;

        coord s0; // inital actor state
        coord terminal_state;

        enum action_space {left, right, up, down, COUNT};

        int n_actions = COUNT;
        int n_states;

        Actor agent;

        std::vector<std::vector<std::string>> grid;
        std::vector<std::vector<double>> Q_table;

        double gamma = 0.9;
        double alpha = 0.1; // decay


        GridWorld(int size, int x0, int y0, int xA, int yA, Actor agent) : 
            size(size), n_states(size*size), agent(agent) {

                s0 = coord(x0, y0);
                terminal_state = coord(xA, yA);

                grid = std::vector<std::vector<std::string>>(size, std::vector<std::string>(size, "-"));

                Q_table = std::vector<std::vector<double>>(size*size, std::vector<double>(n_actions, 0.0));

                grid[terminal_state.second][terminal_state.first] = "A";
        }; 

        GridWorld(Actor agent) : GridWorld(10, 0, 0, 9, 9, agent) {};


        void printGrid() const{
            std::cout << endl;
            std::cout << "grid world" << std::endl;
            std::cout << "-------------------------------------" << std::endl;
            for (auto rit = grid.rbegin(); rit != grid.rend(); ++rit){
                for (const auto& i : *rit){
                    std::cout << i << "   ";
                }
                std::cout << std::endl;
            }
            std::cout << "-------------------------------------" << std::endl;
        }

        // void printQtable() const {
        //     std::cout << endl;
        //     std::cout << "full Q table" << std::endl;
        //     std::cout << "-------------------------------------" << std::endl;
        //     for (const std::vector<double>& row : Q_table) {
        //         for (const auto& i : row) {
        //             std::cout << std::setw(8) << std::fixed << std::setprecision(3) << i << " ";
        //         }
        //         std::cout << std::endl;
        //     }
        //     std::cout << "-------------------------------------" << std::endl;
        // }

        void clearGrid(coord s){
            grid[s.second][s.first] = "-";
            grid[terminal_state.second][terminal_state.first] = "A";
        }

        void updateGrid(coord s, coord s_next) {
            clearGrid(s);
            grid[s_next.second][s_next.first] = "x";
        }


        std::tuple<coord, int, bool> step(coord s, int a){

            coord s_old = s;

            switch (a){
                case left:
                    if (s.first > 0) s.first -= 1;
                    break;
                case right:
                    if (s.first < size-1) s.first += 1;
                    break;
                case up:
                    if (s.second < size-1) s.second += 1;
                    break;
                case down:
                    if (s.second > 0) s.second -= 1;
                    break;
                default:
                    std::cout << "invalid action" << std::endl;
                    break;
            }

            updateGrid(s_old, s);

            if (s == terminal_state) {
                return std::make_tuple(s, 0, true);
            }

            return std::make_tuple(s, -1, false);
        }


        std::tuple<int, double> getQmax(coord s){

            std::vector<double> Q_row = Q_table[ s.first + (s.second * size) ];
            auto max_it = std::max_element(Q_row.begin(), Q_row.end());

            int a_max = std::distance(Q_row.begin(), max_it);
            double Q_max = *max_it;

            return std::make_tuple(a_max, Q_max);
        }

        int epsilonGreedy(coord s){

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis_prob(0.0, 1.0);
            std::uniform_int_distribution<> dis_action(0, 3);

            double prob = dis_prob(gen);

            if (prob < agent.epsilon){
                return dis_action(gen);

            } else {
                return greedy(s);
            }
        }

        int greedy(coord s){
            auto [a, _] = getQmax(s);
            return a;
        }


        template <typename Callable>
        std::tuple<coord, int> run(Callable policy){
            coord s = s0;
            coord s_next;

            int a;
            bool done = false;

            double Q_sa;
            double td_target;
            double td_err;

            int total_reward = 0;

            while (!done) {

                a = policy(s); 

                auto [s_next, r, _done] = step(s, a);
                done = _done;

                Q_sa = Q_table[s.first + s.second * size][a];
                auto [_, Q_next_max] = getQmax(s_next);

                td_target = done ? r : r + gamma * Q_next_max;
                td_err = td_target - Q_sa;

                Q_table[s.first + s.second * size][a] += alpha * td_err;
    
                s = s_next;

                total_reward += r;
            }

            return std::make_pair(s, total_reward);
        }


        void train(int MAX_EPISODES, bool verbose){
            
            for (int i = 0; i < MAX_EPISODES; i++){
                auto [s, tr] = run([this](coord arg) {return epsilonGreedy(arg);});
                if (verbose) printGrid();
                clearGrid(s);
            }
        }

        void inference(int MAX_EPISODES, bool verbose){

            std::filesystem::create_directory("data");
            std::string fname_r = "data/reward.csv";
            std::ofstream file_r(fname_r);

            int LOG_FREQ = 50;
        
            std::vector<int> total_reward_data = std::vector<int>(MAX_EPISODES, 0);
        
            for (int i = 0; i < MAX_EPISODES; i++){

                auto [s, tr] = run([this](coord arg) {return greedy(arg);});

                if (i % LOG_FREQ == 0 || i == MAX_EPISODES - 1) {
                    std::string fname_q = "data/qtable_" + std::to_string(i) + ".csv";
                    std::ofstream file_q(fname_q);
                    // IO and Data operations
                    std::vector<std::vector<double>> Qtable_Grid = convertGreedyQtableToGrid();
                    for (const auto& row : Qtable_Grid){
                        for (const auto& k : row) {
                            file_q << k;
                            if (k < size - 1) file_q << ",";  // comma delimiter
                        }
                        file_q << "\n";  // new line for next row
                    }
                    file_q.close();
                }

                total_reward_data[i] = tr;

                // printGreedyQtableToGrid();
                if (verbose) printGrid();
                clearGrid(s);
            }


            for (const auto& m : total_reward_data){
                file_r << m << ",";
            }
            file_r.close();
        }


    void printGreedyQtableToGrid(){
        std::cout << endl;
        std::cout << "greedy action Q table" << std::endl;
        std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
        std::vector<std::vector<double>> Qtable_Grid = convertGreedyQtableToGrid();
        for (const auto& row : Qtable_Grid){
            for (const auto& i : row) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(3) << i << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
    }

    std::vector<std::vector<double>> convertGreedyQtableToGrid(){

        std::vector<std::vector<double>> Q_tableGrid = std::vector<std::vector<double>>(size, std::vector<double>(size, 0));

        for (int j = 0; j <= size-1; j++){
            for (int i = 0; i <= size-1; i++) {
                auto [_, Q_max] = getQmax(coord(i, j));
                Q_tableGrid[size-1-j][i] = Q_max;
            }
        }
        return Q_tableGrid;
    }
};







int main(){

    Actor agent;

    GridWorld env(agent);

    // env.train(500);
    env.inference(500, true);
    // env.printQtable();
    // env.mapQtableToGrid();
    // env.printGreedyQtableToGrid();

    
    // auto [s1, r1, b1] = env.step(env.s0, env.right);
    // auto [s2, r2, b2] = env.step(s1, env.up);
    // auto [s3, r3, b3] = env.step(s2, env.up);
    // auto [s4, r4, b4] = env.step(s3, env.up);
    // auto [s5, r5, b5] = env.step(s4, env.right);


    // env.printGrid();

    // std::cout<< r5 << std::endl;






    return 0;
}