#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

namespace py = pybind11;

class NeuralNetwork {
public:
    NeuralNetwork(size_t input_size, size_t hidden_size, size_t output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
        // Inicializar pesos con valores aleatorios
        initialize_weights();
    }

    std::vector<double> forward(const std::vector<double>& input) {
        if (input.size() != input_size) {
            throw std::invalid_argument("El tamaño de la entrada no coincide con el tamaño de la capa de entrada.");
        }

        // Propagación hacia adelante de entrada a capa oculta
        hidden_layer_output.clear();
        for (size_t i = 0; i < hidden_size; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < input_size; ++j) {
                sum += input[j] * weights_input_hidden[j][i];
            }
            sum += biases_hidden[i];
            hidden_layer_output.push_back(sigmoid(sum));
        }

        // Propagación hacia adelante de capa oculta a salida
        std::vector<double> output;
        for (size_t i = 0; i < output_size; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < hidden_size; ++j) {
                sum += hidden_layer_output[j] * weights_hidden_output[j][i];
            }
            sum += biases_output[i];
            output.push_back(sigmoid(sum));
        }

        return output;
    }

private:
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    std::vector<double> biases_hidden;
    std::vector<double> biases_output;
    std::vector<double> hidden_layer_output;

    void initialize_weights() {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<> dis(-1.0, 1.0);

        weights_input_hidden.resize(input_size, std::vector<double>(hidden_size));
        for (size_t i = 0; i < input_size; ++i) {
            for (size_t j = 0; j < hidden_size; ++j) {
                weights_input_hidden[i][j] = dis(gen);
            }
        }

        weights_hidden_output.resize(hidden_size, std::vector<double>(output_size));
        for (size_t i = 0; i < hidden_size; ++i) {
            for (size_t j = 0; j < output_size; ++j) {
                weights_hidden_output[i][j] = dis(gen);
            }
        }

        biases_hidden.resize(hidden_size, 0.0);
        biases_output.resize(output_size, 0.0);
        for (size_t i = 0; i < hidden_size; ++i) {
            biases_hidden[i] = dis(gen);
        }
        for (size_t i = 0; i < output_size; ++i) {
            biases_output[i] = dis(gen);
        }
    }

    double sigmoid(double x) const {
        return 1.0 / (1.0 + std::exp(-x));
    }
};

PYBIND11_MODULE(neural_net, m) {
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<size_t, size_t, size_t>())
        .def("forward", &NeuralNetwork::forward);
}
