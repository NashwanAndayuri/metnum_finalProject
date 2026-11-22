#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include <cctype>
using namespace std;
namespace fs = std::filesystem;

#define h 1.e-5
#define lr 0.05

vector<array<double, 3>> trainData, testData; // scaled pitch, scaled ZCR, label

double a = 0.1, b = 0.1, c = 0.1;
double dEa = 1, dEb = 1, dEc = 1;

// x1 = pitch / 300
// x2 = zero-crossing rate * 10
double f(double x1, double x2)
{
    return a * x1 + b * x2 + c;
}

// Average squared error on the training data
double E()
{
    double sum = 0;
    for (auto row : trainData) {
        double predicted = f(row[0], row[1]);
        sum += pow(row[2] - predicted, 2);
    }
    return sum / static_cast<double>(trainData.size());
}

// Estimate each derivative using central differences
double centralDifference(double& coefficient)
{
    double original = coefficient;
    coefficient = original + h;
    double errorPlus = E();
    coefficient = original - h;
    double errorMinus = E();
    coefficient = original;
    return (errorPlus - errorMinus) / (2 * h);
}

void derivative_E()
{
    dEa = centralDifference(a);
    dEb = centralDifference(b);
    dEc = centralDifference(c);
}

// Read 16 kHz 16-bit PCM WAV files
unsigned int readNumber(ifstream& file, int bytes)
{
    unsigned int value = 0;
    for (int i = 0; i < bytes; i++) {
        int ch = file.get();
        if (ch == EOF) return 0;
        value |= (unsigned int)(unsigned char)ch << (8 * i);
    }
    return value;
}

vector<double> readWav(const fs::path& path)
{
    ifstream file(path, ios::binary);
    char id[4];
    if (!file.read(id, 4) || string(id, 4) != "RIFF") return {};
    file.seekg(8);
    if (!file.read(id, 4) || string(id, 4) != "WAVE") return {};

    unsigned int format = 0, channels = 0, rate = 0, bits = 0;
    vector<unsigned char> bytes;
    while (file.read(id, 4)) {
        unsigned int size = readNumber(file, 4);
        if (!file || size > 100000000) return {};
        auto next = file.tellg() + streamoff(size + size % 2);
        if (string(id, 4) == "fmt " && size >= 16) {
            format = readNumber(file, 2);
            channels = readNumber(file, 2);
            rate = readNumber(file, 4);
            readNumber(file, 4);
            readNumber(file, 2);
            bits = readNumber(file, 2);
        } else if (string(id, 4) == "data") {
            bytes.resize(size);
            if (!file.read((char*)bytes.data(), size)) return {};
        }
        if (!file) return {};
        file.seekg(next);
        if (!file) return {};
    }
    if (format != 1 || rate != 16000 || bits != 16 ||
        (channels != 1 && channels != 2) || bytes.empty() ||
        bytes.size() % (channels * 2) != 0) return {};

    vector<double> audio;
    for (size_t i = 0; i + channels * 2 <= bytes.size(); i += channels * 2) {
        double sum = 0;
        for (unsigned int j = 0; j < channels; j++) {
            size_t p = i + j * 2;
            int sample = bytes[p] + 256 * bytes[p + 1];
            if (sample >= 32768) sample -= 65536;
            sum += sample / 32768.0;
        }
        audio.push_back(sum / channels);
    }
    return audio;
}

// Get pitch and zerocrossing rate from the audio
vector<double> getFeatures(vector<double> audio)
{
    if (audio.size() < 4000) return {};

    // Reduce low-frequency noise
    double previousX = 0, previousY = 0;
    for (double& x : audio) {
        double y = 0.995 * (previousY + x - previousX);
        previousX = x;
        previousY = y;
        x = y;
    }

    // Estimate pitch in three parts of the recording
    const int windowSize = 1024, minLag = 40, maxLag = 240;
    double pitchSum = 0;
    int valid = 0;
    for (int k = 1; k <= 3; k++) {
        int start = int(audio.size() - windowSize) * k / 4;
        array<double, maxLag + 2> correlation{};
        for (int lag = minLag - 1; lag <= maxLag + 1; lag++) {
            double ab = 0, aa = 0, bb = 0;
            for (int i = 0; i < windowSize - lag; i++) {
                double x = audio[start + i];
                double y = audio[start + i + lag];
                ab += x * y;
                aa += x * x;
                bb += y * y;
            }
            correlation[lag] = ab / sqrt(aa * bb + 1.e-12);
        }

        double best = *max_element(correlation.begin() + minLag,
                                   correlation.begin() + maxLag + 1);
        double minPeak = max(0.5, 0.95 * best);
        // A later peak can cover more than one period
        for (int lag = minLag; lag <= maxLag; lag++) {
            if (correlation[lag] >= minPeak &&
                correlation[lag] >= correlation[lag - 1] &&
                correlation[lag] > correlation[lag + 1]) {
                pitchSum += 16000.0 / lag;
                valid++;
                break;
            }
        }
    }
    if (valid == 0) return {};
    double x1 = (pitchSum / valid) / 300.0;

    // Count sign changes in the middle half of the recording
    int start = int(audio.size()) / 4;
    int end = int(audio.size()) * 3 / 4;
    int crossings = 0;
    for (int i = start + 1; i < end; i++)
        if ((audio[i] >= 0) != (audio[i - 1] >= 0)) crossings++;
    double x2 = 10.0 * crossings / (end - start);
    return {x1, x2};
}

// Use label 0 for odd actor numbers and 1 for even ones
void loadData(const fs::path& folder, int& trainSkipped, int& testSkipped)
{
    for (const auto& entry : fs::recursive_directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        string name = entry.path().filename().string();
        if (name.size() < 10 || name.substr(0, 6) != "03-01-" ||
            name.substr(name.size() - 4) != ".wav") continue;
        char n1 = name[name.size() - 6], n2 = name[name.size() - 5];
        if (!isdigit((unsigned char)n1) || !isdigit((unsigned char)n2)) continue;
        int actor = (n1 - '0') * 10 + (n2 - '0');
        if (actor < 1 || actor > 24) continue;

        vector<double> x = getFeatures(readWav(entry.path()));
        int& skipped = actor <= 16 ? trainSkipped : testSkipped;
        if (x.empty()) {
            skipped++;
            continue;
        }
        array<double, 3> row = {x[0], x[1], actor % 2 == 0 ? 1.0 : 0.0};
        if (actor <= 16) trainData.push_back(row);
        else testData.push_back(row); // Actors 17 to 24
    }
}

int main(int argc, char* argv[])
{
    fs::path dataset = argc > 1 ? argv[1] : "dataset";
    if (!fs::is_directory(dataset)) {
        cout << "Dataset folder not found: " << dataset << '\n';
        cout << "Run from the project folder or pass the dataset path.\n";
        return 1;
    }
    cout << "Loading RAVDESS...\n";
    int trainSkipped = 0, testSkipped = 0;
    loadData(dataset, trainSkipped, testSkipped);
    if (trainData.empty() || testData.empty()) {
        cout << "No usable WAV files. Need all Actor_01...Actor_24 folders.\n";
        return 1;
    }
    cout << "Training samples: " << trainData.size()
         << " (skipped " << trainSkipped << ")\n";
    cout << "Testing samples: " << testData.size()
         << " (skipped " << testSkipped << ")\n";
    cout << "Initial MSE: " << E() << '\n';

    // Use gradient descent to update a, b, and c
    long iter = 0;
    while (iter < 5000) {
        derivative_E();
        if (sqrt(dEa*dEa + dEb*dEb + dEc*dEc) < 1.e-5) break;
        a = a - dEa * lr;
        b = b - dEb * lr;
        c = c - dEc * lr;
        iter++;
        if (iter % 500 == 0)
            cout << "Iteration " << iter << " MSE: " << E() << '\n';
        if (!isfinite(E())) {
            cout << "Training diverged. Try a smaller learning rate.\n";
            return 1;
        }
    }
    cout << "Final MSE: " << E() << '\n';
    cout << "Iterations: " << iter << " | a=" << a
         << " b=" << b << " c=" << c << '\n';

    // Count correct predictions on the test set
    int correct = 0;
    for (auto row : testData) {
        int guess = f(row[0], row[1]) >= 0.5 ? 1 : 0;
        if (guess == int(row[2])) correct++;
    }
    cout << "Test accuracy: " << 100.0 * correct / static_cast<double>(testData.size())
         << "% (" << correct << '/' << testData.size() << " classified clips)\n";

    // Try one WAV file or press Enter to quit
    cout << "Enter a 16 kHz WAV file path (or press Enter to exit): ";
    string path;
    getline(cin, path);
    if (path.size() >= 2 && path.front() == '"' && path.back() == '"')
        path = path.substr(1, path.size() - 2);
    if (!path.empty()) {
        vector<double> x = getFeatures(readWav(path));
        if (x.empty()) cout << "Cannot read or analyze this WAV file.\n";
        else {
            double result = f(x[0], x[1]);
            cout << "Regression score: " << result << '\n';
            cout << "Predicted RAVDESS label: "
                 << (result >= 0.5 ? "female-labeled" : "male-labeled") << '\n';
            cout << "This prediction refers to the RAVDESS dataset labels.\n";
        }
    }
}
