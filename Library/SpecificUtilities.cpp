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