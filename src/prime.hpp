#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <cmath>
#include <functional>
#include <iostream>
#include <unordered_set>
#pragma once

using namespace std;
class Value;
using ValuePtr = shared_ptr<Value>;

class Value : public enable_shared_from_this<Value> {
private:
    inline static size_t currentID = 0;
    float data;
    float grad;
    string op;
    size_t id;
    vector<ValuePtr> prev;

    void topo_sort(vector<ValuePtr>& topo, unordered_set<Value*>& visited);

public:
    Value(float data, const string &op, size_t id);
    ~Value();

    static ValuePtr create(float data, const string& op = "");

    void print(bool verbose = true, int depth = 0);

    float get_val();
    void set_prev(const vector<ValuePtr>& parents);
    void set_grad(float grad);
    string get_op();

    float get_grad();
    void add_grad(float g);

    static ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr sub(const ValuePtr& lhs, const ValuePtr& rhs);
    static ValuePtr mult(const ValuePtr& lhs, const ValuePtr& rhs);
    function<void()> _backward;
    void backward();
};