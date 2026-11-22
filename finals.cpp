#include <iostream>
#include <cmath>
#define h 1.e-10
#define lr 0.1

using namespace std;

double data[3][3] = {
    {0.33, 0.9, 0.9},
    {0.29, 0.8, 0.7},
    {0.25, 0.95, 0.99}
};

double a, b, c;
double dEa = 1, dEb = 1, dEc = 1;

// Hard-coded linear model: y = a*x1 + b*x2 + c
double f(double x1, double x2)
{
    return a * x1 + b * x2 + c;
}

double E()
{
    double sum = 0;
    int i;

    for (i = 0; i < 3; i++)
    {
        double predicted = f(data[i][0], data[i][1]);
        sum += pow(data[i][2] - predicted, 2);
    }

    return sum;
}

void derivative_E()
{
    double temp, e_temp1, e_temp2;

    temp = a;
    a = a + h;
    e_temp1 = E();
    a = temp;
    a = a - h;
    e_temp2 = E();
    dEa = (e_temp1 - e_temp2) / (2 * h);
    a = temp;

    temp = b;
    b = b + h;
    e_temp1 = E();
    b = temp;
    b = b - h;
    e_temp2 = E();
    dEb = (e_temp1 - e_temp2) / (2 * h);
    b = temp;

    temp = c;
    c = c + h;
    e_temp1 = E();
    c = temp;
    c = c - h;
    e_temp2 = E();
    dEc = (e_temp1 - e_temp2) / (2 * h);
    c = temp;
}

int main()
{
    a = 0.1;
    b = 0.1;
    c = 0.1; // initialization of weights

    long iter = 0;
    while (sqrt(pow(dEa, 2) + pow(dEb, 2) + pow(dEc, 2)) > 1.e-5 && iter < 1000000)
    {
        derivative_E();
        a = a - (dEa * lr);
        b = b - (dEb * lr);
        c = c - (dEc * lr);
        iter++;

        if (iter % 1000 == 0)
            cout << iter << " dEa: " << dEa << " dEb: " << dEb << " dEc: " << dEc << '\n';
    }

    cout << "total loop: " << iter << " iterations: a=" << a << " b=" << b << " c=" << c << '\n';
}