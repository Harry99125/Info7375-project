#include <stdio.h>
#include <math.h>
double sigmoid(double x) {
   // printf("%f/n",1.0 / (1.0 + exp(-x)));
    return 1.0 / (1.0 + exp(-x));
}


typedef struct {
    double weights[2];
    double bias;       
} Neuron;


double feedforward(Neuron *n, double inputs[2]) {
    int i;
    double total=0.0;
    for(i=0;i<2;i++){
     total+=n->weights[i]*inputs[i];
    }
    total+=n->bias;
    return sigmoid(total);
}
double feedforward2(Neuron *n,double x[2]){
    double out_h1=feedforward(n,x);

      double out_h2=feedforward(n,x);
       printf("%f\n",out_h2);
      double o1[2] = {out_h1, out_h2};
        double out_o1=feedforward(n,o1);
        return out_o1;
}


int main() {
    Neuron n;
    n.weights[0] = 0.0;
    n.weights[1] = 1.0;
    n.bias = 0.0;
    


    double inputs[2] = {2.0, 3.0};
    double output = feedforward2(&n, inputs);
    printf("%f\n", output); 
    return 0;
}