#pragma once

#include <iostream>
#include <cassert>
#include <iomanip>
#include <vector>
#include <string>

void display(const std::string& msg);
void display(const std::string& label, const std::vector<double>& v);
void displayStats(const double d, const double pass);