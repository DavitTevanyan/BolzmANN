#ifndef SPECIFIC_UTILITIES_H
#define SPECIFIC_UTILITIES_H

#include <iostream>
#include <cassert>
#include <iomanip>
#include <vector>
#include <string>

const bool alignRight = true;

void display(const std::string& label, const std::vector<double>& v, bool alignRight = false);
void displayNetError(const double d);

#endif // SPECIFIC_UTILITIES_H
