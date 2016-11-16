#include "SpecificUtilities.h"

void display(const std::string& label, const std::vector<double>& v, bool alignRight)
{
    if (alignRight)
        std::cout << std::right << std::setw(30);

    std::cout << label << " ";

    for (const auto& elem : v)
        std::cout << elem << " ";

    std::cout << std::endl;
}

void displayStats(const double error, const double pass)
{
    // Report how well the training is working, average over recent samples
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "              Pass: " << pass        << std::endl;
    std::cout << " Ann average error: " << error       << std::endl;
    std::cout << "-----------------------------------" << std::endl;
}