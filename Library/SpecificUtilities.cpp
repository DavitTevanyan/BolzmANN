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

void displayNetError(const double error)
{
    // Report how well the training is working, average over recent samples
    std::cout << "-----------------------------------" << std::endl;
    std::cout << "Ann recent average error: " << error << std::endl;
    std::cout << "-----------------------------------" << std::endl;
}