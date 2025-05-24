#include <stdio.h>
#include <math.h>
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}


typedef struct {
    double weights[2];
    double bias;       
} Neuron;


double feedforward(Neuron *n, double inputs[2]) {
    int i;
    double total;
    for(i=0;i<2;i++){
     total+=n->weights[i]*inputs[i];
    }
    total+=n->bias;
    return sigmoid(total);
}

int main() {
    Neuron n;
    n.weights[0] = 0.0;
    n.weights[1] = 1.0;
    n.bias = 4.0;

    double inputs[2] = {2.0, 3.0};
    double output = feedforward(&n, inputs);
    printf("%f\n", output); 
    return 0;
}