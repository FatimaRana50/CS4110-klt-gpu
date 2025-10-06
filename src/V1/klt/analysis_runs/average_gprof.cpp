#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <map>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

struct FunctionData {
    double pct_sum = 0.0;
    double self_sum = 0.0;
    double total_sum = 0.0;
    int count = 0;
};

string normalizeName(string name) {
    // Remove leading underscores
    name = regex_replace(name, regex("^_+"), "");
    return name;
}

map<string, tuple<double, double, double>> parseGprofFile(const string &filename) {
    map<string, tuple<double, double, double>> data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << " Error opening file: " << filename << endl;
        return data;
    }

    string line;
    // Example line:
    //  38.89      0.07     0.07       63     1.11     1.11  _convolveImageHoriz
    regex pattern(R"(\[\d+\]\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)(?:\s+\S+){0,3}\s+([A-Za-z0-9_:+~]+)\s+\[\d+\])");

    smatch match;

    while (getline(file, line)) {
        if (regex_search(line, match, pattern)) {
            double pct_time = stod(match[1]);
            double self_sec = stod(match[2]);
            double total_sec = stod(match[3]);
            string func = normalizeName(match[4]);
            data[func] = make_tuple(pct_time, self_sec, total_sec);
        }
    }
    return data;
}

map<string, tuple<double, double, double>> averageGprof(const vector<string> &files) {
    map<string, FunctionData> sums;

    for (const auto &file : files) {
        auto parsed = parseGprofFile(file);
        for (const auto &[func, vals] : parsed) {
            auto [pct, self, total] = vals;
            sums[func].pct_sum += pct;
            sums[func].self_sum += self;
            sums[func].total_sum += total;
            sums[func].count++;
        }
    }

    map<string, tuple<double, double, double>> averages;
    for (const auto &[func, s] : sums) {
        averages[func] = make_tuple(
            s.pct_sum / s.count,
            s.self_sum / s.count,
            s.total_sum / s.count
        );
    }
    return averages;
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << "Usage: ./average_gprof run1.txt run2.txt run3.txt" << endl;
        return 1;
    }

    vector<string> files;
    for (int i = 1; i < argc; i++) files.push_back(argv[i]);

    auto results = averageGprof(files);

    string output_file = fs::path(files[0]).parent_path().string() + "averaged_results.txt";
    ofstream out(output_file);
    if (!out.is_open()) {
        cerr << "Cannot write to " << output_file << endl;
        return 1;
    }

    out << left << setw(35) << "Function"
        << setw(15) << "Avg % Time"
        << setw(20) << "Avg Self(s)"
        << setw(20) << "Avg Total(s)" << "\n";
    out << string(90, '=') << "\n";

    vector<pair<string, tuple<double, double, double>>> sorted(results.begin(), results.end());
    sort(sorted.begin(), sorted.end(), [](auto &a, auto &b) {
        return get<0>(a.second) > get<0>(b.second); // Sort by % time
    });

    cout << "\n Top Hotspot Functions (by % time):\n";
    int limit = min(10, (int)sorted.size());
    for (int i = 0; i < limit; i++) {
        cout << setw(25) << left << sorted[i].first
             << " %time_avg=" << fixed << setprecision(2) << get<0>(sorted[i].second)
             << "% self_avg=" << get<1>(sorted[i].second)
             << "s total_avg=" << get<2>(sorted[i].second) << "s\n";
    }

    for (auto &[func, vals] : sorted) {
        out << left << setw(35) << func
            << setw(15) << fixed << setprecision(2) << get<0>(vals)
            << setw(20) << get<1>(vals)
            << setw(20) << get<2>(vals) << "\n";
    }

    cout << "\n Results saved to: " << output_file << endl;
    return 0;
}
