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
using stringGrid = std::vector<std::vector<std::string>>;
using doubleGrid = std::vector<std::vector<double>>;



class Actor {

    public:

        double epsilon; // exploration fraction

        Actor(double epsilon) : epsilon(epsilon) {};

        Actor() : Actor(0.1) {};    
};

class GridWorld {

    public:

    using coord = std::pair<int, int>;

        int size_x;
        int size_y;

        coord s0; // inital agent state
        coord terminal_state;

        enum action_space {left, right, up, down, COUNT};

        int n_actions = COUNT;
        int n_states;

        Actor agent;

        stringGrid grid;
        doubleGrid Q_table;

        double gamma = 0.9;
        double alpha = 0.1; // decay


        GridWorld(int size_x, int size_y, int x0, int y0, int xA, int yA, Actor agent) : 
            size_x(size_x), size_y(size_y), n_states(size_x * size_y), agent(agent) {

                s0 = coord(x0, y0);
                terminal_state = coord(xA, yA);

                grid = stringGrid(size_x, std::vector<std::string>(size_y, "-"));

                Q_table = doubleGrid(n_states, std::vector<double>(n_actions, 0.0));

                grid[terminal_state.first][terminal_state.second] = "A";
        };


        GridWorld(Actor agent) : GridWorld(10, 10, 0, 0, 9, 9, agent) {};


        void printGrid() const{
            std::cout << std::endl;
            std::cout << "grid world" << std::endl;
            std::cout << "-------------------------------------" << std::endl;

            for (int j = size_y-1; j >= 0; j--){
                for (int i = 0; i < size_x; i++){
                    std::cout << grid[i][j] << "   ";
                }
                std::cout << std::endl;
            }
            std::cout << "-------------------------------------" << std::endl;
        }
  

        void clearGrid(coord s){
            grid[s.first][s.second] = "-";
            grid[terminal_state.first][terminal_state.second] = "A";
        }


        void updateGrid(coord s, coord s_next) {
            clearGrid(s);
            grid[s_next.first][s_next.second] = "x";
        }


        std::tuple<coord, int, bool> step(coord s, int a){

            coord s_old = s;

            switch (a){
                case left:
                    if (s.first > 0) s.first -= 1;
                    break;
                case right:
                    if (s.first < size_x - 1) s.first += 1;
                    break;
                case up:
                    if (s.second > 0) s.second -= 1;
                    break;
                case down:
                    if (s.second < size_y - 1) s.second += 1;
                    break;
                default:
                    std::cout << "invalid action" << std::endl;
                    break;
            }

            updateGrid(s_old, s);

            if (s == terminal_state) return std::make_tuple(s, 0, true);

            return std::make_tuple(s, -1, false);
        }


        std::tuple<int, double> getQmax(coord s){

            std::vector<double> Q_row = Q_table[ s.first + s.second * size_x ];
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
        std::tuple<coord, int, int> run(Callable policy){
            coord s = s0;
            coord s_next;

            int a;
            bool done = false;

            double Q_sa;
            double td_target;
            double td_err;

            int total_reward = 0;
            int episode_length = 0;

            int MAX_STEPS = 1000;

            while (!done && episode_length < MAX_STEPS) {

                a = policy(s);

                auto [s_next, r, _done] = step(s, a);
                done = _done;
                
                Q_sa = Q_table[s.first + s.second * size_x][a];
                auto [_, Q_next_max] = getQmax(s_next);

                td_target = done ? r : r + gamma * Q_next_max;
                td_err = td_target - Q_sa;

                Q_table[s.first + s.second * size_x][a] += alpha * td_err;
    
                s = s_next;

                total_reward += r;
                episode_length++;
                // printGrid();
            }

            return std::make_tuple(s, total_reward, episode_length);
        }


        void train(int MAX_EPISODES, bool verbose){

            std::filesystem::create_directory("data");

            int LOG_FREQ = 50;
        
            std::vector<int> total_reward_data = std::vector<int>(MAX_EPISODES, 0);
            std::vector<int> episode_len_data = std::vector<int>(MAX_EPISODES, 0);
            
            for (int i = 0; i < MAX_EPISODES; i++){
                auto [s, tr, ep_len] = run([this](coord arg) {return epsilonGreedy(arg);});

                // IO operations
                if (i % LOG_FREQ == 0 || i == MAX_EPISODES - 1) saveQtableGridToFile(i);
                    
                total_reward_data[i] = tr;
                episode_len_data[i] = ep_len;

                if (verbose) { printGrid(); printGreedyQtableGrid(); }
                clearGrid(s);
            }

            // IO operations
            saveVectorToFile(total_reward_data, "total_reward");
            saveVectorToFile(episode_len_data, "episode_len");
        }


        void inference(bool verbose){
        
            auto [s, _, __] = run([this](coord arg) {return greedy(arg);});

            if (verbose) { printGrid(); printGreedyQtableGrid(); }
            clearGrid(s);

        }


        void saveQtableGridToFile(int episode){

            std::string fname_q = "data/qtable_" + std::to_string(episode) + ".csv";
            std::ofstream file_q(fname_q);
            doubleGrid Qtable_Grid = convertGreedyQtableToGrid();

            for (int j = size_y-1; j >= 0; j--){
                for (int i = 0; i < size_x; i++){
                    file_q << Qtable_Grid[i][j];
                    if (i < size_x - 1) file_q << ",";
                }
                file_q << "\n";
            }
            file_q.close();
        }


        void saveVectorToFile(std::vector<int>& data, std::string fname){
            std::string path_name = "data/" + fname + ".csv";
            std::ofstream file_v(path_name);
            for (size_t i = 0; i < data.size(); i++){
                file_v << data[i];
                if (i < data.size() - 1) file_v << ",";  // comma delimiter
            }
            file_v.close();
        }


        void printGreedyQtableGrid(){

            doubleGrid Qtable_Grid = convertGreedyQtableToGrid();

            std::cout << std::endl;
            std::cout << "greedy action Q table" << std::endl;
            std::cout << "-----------------------------------------------------------------------------------------" << std::endl;

            for (int j = size_y-1; j >= 0; j--){
                for (int i = 0; i < size_x; i++){
                    std::cout << std::setw(8) << std::fixed << std::setprecision(3) << Qtable_Grid[i][j] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "-----------------------------------------------------------------------------------------" << std::endl;
        }


        doubleGrid convertGreedyQtableToGrid(){

            doubleGrid Q_tableGrid(size_x, std::vector<double>(size_y, 0));

            for (int j = 0; j < size_y; j++) {
                for (int i = 0; i < size_x; i++){
                    auto [_, Q_max] = getQmax(coord(i, j));
                    Q_tableGrid[i][j] = Q_max;
                }
            }
            return Q_tableGrid;
        }

};





int main(){

    Actor agent(0.1);
    // Actor agent();

    int size_x = 10;
    int size_y = 10;
    int x0 = 0;
    int y0 = 0;
    int xA = size_x - 1;
    int yA = size_y - 1;

    GridWorld env(size_x, size_y, x0, y0, xA, yA, agent);
    // GridWorld env(agent);

    env.train(1000, false);
    env.inference(true);






    return 0;
}