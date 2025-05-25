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
      Neuron n1;
    n1.weights[0] = 1.0;
    n1.weights[1] = 2.0;
    n1.bias = 2.0;
      Neuron n2;
    n2.weights[0] = 3.0;
    n2.weights[1] = 2.0;
    n2.bias = 4.0;

    double inputs[2] = {2.0, 3.0};
       double inputs1[2] = {1.0, 4.0};
          double inputs2[2] = {4.0, 2.0};
    double output = feedforward(&n, inputs);
    double output1 = feedforward(&n1, inputs1);
    double output2 = feedforward(&n2, inputs2);
    printf("%f test case 1\n", output); 
      printf("%f test case 2\n", output1); 
        printf("%f test case 3\n", output2); 
    return 0;
}