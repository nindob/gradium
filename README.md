# Gradium 
A neural network framework from scratch in C++ to understand the fundamentals of automatic differentiation and computational graphs.

- [Current] Micrograd Scalar Backprop and Layers with Toposort
    - Operator Overloading
    - Toposort for Reduced Compile Time
    - End to End Train-Test Loop with AdamW
- [Next Steps]
    - Vector & Matrix Support
    - Operator Fusion
    - Computational Graph Optimization

To Run:
g++ -std=c++20 main.cpp prime.cpp loss.cpp mat.cpp -o gradium
./gradium