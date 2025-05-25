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
  
      double o1[2] = {out_h1, out_h2};
        double out_o1=feedforward(n,o1);
        return out_o1;
}


int main() {
    Neuron n;
    n.weights[0] = 0.0;
    n.weights[1] = 1.0;
    n.bias = 0.0;
    Neuron n1;
     n1.weights[0] = 1.0;
    n1.weights[1] = 1.0;
    n1.bias = 0.0;
        Neuron n2;
     n2.weights[0] = 0.5;
    n2.weights[1] = 0.5;
    n2.bias = 1.0;


    double inputs[2] = {2.0, 3.0};
     double inputs1[2] = {1.0, 1.0};
       double inputs2[2] = {2.0, 4.0};
    double output = feedforward2(&n, inputs);//test case 1
    double output1 = feedforward2(&n1, inputs1);// test case 2
       double output2 = feedforward2(&n2, inputs2);// test case 2
    printf("%f test case 1\n", output); 
       printf("%f test case 2\n", output1); 
          printf("%f test case 3\n", output2); 
    return 0;
}
