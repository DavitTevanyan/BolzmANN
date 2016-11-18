#ifndef SPECIFIC_UTILITIES_H
#define SPECIFIC_UTILITIES_H

#include <iostream>
#include <cassert>
#include <iomanip>
#include <vector>
#include <string>

const bool alignRight = true;

void display(const std::string& msg);
void display(const std::string& label, const std::vector<double>& v);
void displayStats(const double d, const double pass);

#endif // SPECIFIC_UTILITIES_H
